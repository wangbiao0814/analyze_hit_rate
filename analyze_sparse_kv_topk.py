#!/usr/bin/env python3
"""Replay sparse-KV top-k logs and compare CPU-side cache hit rates.

The input is the eager-mode log emitted by ``SparseKVCacheManager``::

    SPARSE_KV_DECODE_TOPK rank=0 rid=req-1 req_pool_idx=3 \
        layer_id=5 decode_round=1 seq_len=4097 topk=[...]

State is isolated by ``(rank, rid, layer_id)``. Negative top-k padding values
are ignored; every non-negative token is one cache lookup in the hit-rate
denominator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    Tuple,
)


POLICY_CURRENT_TOPK = "current_topk_2048"
POLICY_TOPK_ORDER = "topk_order_lru_4096"
POLICY_HIT_MISS_ORDER = "hit_miss_order_lru_4096"
POLICY_AGE_PROBATION = "age_probation_lru_4096"
POLICY_NAMES = (
    POLICY_CURRENT_TOPK,
    POLICY_TOPK_ORDER,
    POLICY_HIT_MISS_ORDER,
    POLICY_AGE_PROBATION,
)
STAMP_MAX = (1 << 24) - 1

LOG_PATTERN = re.compile(
    r"SPARSE_KV_DECODE_TOPK\s+"
    r"rank=(?P<rank>-?\d+)\s+"
    r"rid=(?P<rid>.*?)\s+"
    r"req_pool_idx=(?P<req_pool_idx>-?\d+)\s+"
    r"layer_id=(?P<layer_id>-?\d+)\s+"
    r"decode_round=(?P<decode_round>\d+)\s+"
    r"seq_len=(?P<seq_len>\d+)\s+"
    r"topk=(?P<topk>\[.*\])\s*$"
)


@dataclass(frozen=True)
class TopKRecord:
    rank: int
    rid: str
    req_pool_idx: int
    layer_id: int
    decode_round: int
    seq_len: int
    topk: Tuple[int, ...]

    @property
    def request_key(self) -> str:
        if self.rid != "unknown":
            return self.rid
        return "unknown@req_pool_idx={}".format(self.req_pool_idx)

    @property
    def stream_key(self) -> Tuple[int, str, int]:
        return (self.rank, self.request_key, self.layer_id)


@dataclass
class HitRateStats:
    hits: int = 0
    misses: int = 0
    rounds: int = 0

    def add(self, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses
        self.rounds += 1

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses


class CachePolicy(Protocol):
    def access(self, topk: Sequence[int]) -> Tuple[int, int]: ...


class CurrentTopKPolicy:
    """Capacity-2048 baseline: every round replaces the cache with its top-k."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tokens = set()

    def access(self, topk: Sequence[int]) -> Tuple[int, int]:
        if len(topk) > self.capacity:
            raise ValueError(
                "valid top-k length {} exceeds strategy-1 capacity {}".format(
                    len(topk), self.capacity
                )
            )
        hits = sum(token in self.tokens for token in topk)
        self.tokens = set(topk)
        return hits, len(topk) - hits


class TopKOrderLRUPolicy:
    """Move the current top-k to the MRU tail in top-k order."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.queue: OrderedDict[int, None] = OrderedDict()

    def access(self, topk: Sequence[int]) -> Tuple[int, int]:
        hits = sum(token in self.queue for token in topk)
        for token in topk:
            if token in self.queue:
                del self.queue[token]
            self.queue[token] = None
        while len(self.queue) > self.capacity:
            self.queue.popitem(last=False)
        return hits, len(topk) - hits

    def snapshot(self) -> Tuple[int, ...]:
        return tuple(self.queue)


class StableHitMissLRUPolicy:
    """Stable hit/miss queue matching timestamp LRU with probation age zero.

    Cache order is oldest to youngest. Non-hits stay at the head, hits move to
    the tail in their previous cache order, and misses follow in top-k order.
    Empty physical slots are represented explicitly so cold-start behavior
    matches the fixed-size device metadata.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.queue: List[Optional[int]] = [None] * capacity

    def access(self, topk: Sequence[int]) -> Tuple[int, int]:
        resident = {token for token in self.queue if token is not None}
        requested = set(topk)
        hits = [token for token in topk if token in resident]
        misses = [token for token in topk if token not in resident]

        non_hits_in_cache_order = [
            token for token in self.queue if token is None or token not in requested
        ]
        hits_in_cache_order = [
            token for token in self.queue if token is not None and token in requested
        ]
        refreshed = non_hits_in_cache_order + hits_in_cache_order
        if len(misses) > len(refreshed):
            raise ValueError("miss count exceeds cache capacity")
        self.queue = refreshed[len(misses) :] + misses
        return len(hits), len(misses)

    def snapshot(self) -> Tuple[int, ...]:
        return tuple(token for token in self.queue if token is not None)


@dataclass
class _AgeEntry:
    token: Optional[int]
    age: int


class AgeProbationLRUPolicy:
    """CPU reference for the fixed-slot timestamp-age probation policy."""

    def __init__(self, capacity: int, probation_age: int) -> None:
        if probation_age < 0 or probation_age > STAMP_MAX:
            raise ValueError(
                "probation_age must be in [0, {}], got {}".format(
                    STAMP_MAX, probation_age
                )
            )
        self.capacity = capacity
        self.probation_age = probation_age
        self.entries = [_AgeEntry(None, 0) for _ in range(capacity)]

    def access(self, topk: Sequence[int]) -> Tuple[int, int]:
        token_to_entry = {
            entry.token: entry for entry in self.entries if entry.token is not None
        }
        hit_tokens = [token for token in topk if token in token_to_entry]
        miss_tokens = [token for token in topk if token not in token_to_entry]
        hit_set = set(hit_tokens)

        non_hit_entries = []
        hit_entries = []
        for entry in self.entries:
            entry.age = min(entry.age, STAMP_MAX - 1) + 1
            if entry.token in hit_set:
                entry.age = 0
                hit_entries.append(entry)
            else:
                non_hit_entries.append(entry)

        # The input is already in stable descending-age order. Incrementing all
        # ages preserves that order, while hit entries move stably to age zero.
        self.entries = non_hit_entries + hit_entries
        if len(miss_tokens) > self.capacity - len(hit_tokens):
            raise ValueError("miss count would evict a hit from the same round")

        victims = self.entries[: len(miss_tokens)]
        survivors = self.entries[len(miss_tokens) :]
        for victim, token in zip(victims, miss_tokens):
            victim.token = token
            victim.age = self.probation_age

        # Insert the victim block after existing entries of the same age. This
        # is the linear-time equivalent of the stable descending sort in the
        # kernel CPU reference.
        insertion_index = len(survivors)
        for index, entry in enumerate(survivors):
            if entry.age < self.probation_age:
                insertion_index = index
                break
        self.entries = (
            survivors[:insertion_index]
            + victims
            + survivors[insertion_index:]
        )
        return len(hit_tokens), len(miss_tokens)

    def snapshot(self) -> Tuple[int, ...]:
        return tuple(
            entry.token for entry in self.entries if entry.token is not None
        )


def parse_log_line(line: str) -> Optional[TopKRecord]:
    if "SPARSE_KV_DECODE_TOPK" not in line:
        return None
    match = LOG_PATTERN.search(line)
    if match is None:
        raise ValueError("malformed SPARSE_KV_DECODE_TOPK line")

    parsed_topk = ast.literal_eval(match.group("topk"))
    if not isinstance(parsed_topk, list) or not all(
        isinstance(token, int) for token in parsed_topk
    ):
        raise ValueError("topk must be a Python list of integers")

    return TopKRecord(
        rank=int(match.group("rank")),
        rid=match.group("rid"),
        req_pool_idx=int(match.group("req_pool_idx")),
        layer_id=int(match.group("layer_id")),
        decode_round=int(match.group("decode_round")),
        seq_len=int(match.group("seq_len")),
        topk=tuple(parsed_topk),
    )


class HitRateAnalyzer:
    def __init__(
        self,
        topk_capacity: int = 2048,
        device_capacity: int = 4096,
        probation_age: int = 4,
    ) -> None:
        if topk_capacity <= 0 or device_capacity <= 0:
            raise ValueError("cache capacities must be positive")
        if topk_capacity > device_capacity:
            raise ValueError("topk capacity cannot exceed device capacity")
        self.topk_capacity = topk_capacity
        self.device_capacity = device_capacity
        self.probation_age = probation_age
        self._policies: Dict[
            Tuple[int, str, int], Dict[str, CachePolicy]
        ] = {}
        self._last_round: Dict[Tuple[int, str, int], Tuple[int, int]] = {}
        self.overall = self._new_stats()
        self.by_layer: Dict[Tuple[int, int], Dict[str, HitRateStats]] = {}
        self.by_request: Dict[Tuple[int, str], Dict[str, HitRateStats]] = {}
        self.by_request_layer: Dict[
            Tuple[int, str, int], Dict[str, HitRateStats]
        ] = {}

    @staticmethod
    def _new_stats() -> Dict[str, HitRateStats]:
        return {name: HitRateStats() for name in POLICY_NAMES}

    def _new_policies(self) -> Dict[str, CachePolicy]:
        return {
            POLICY_CURRENT_TOPK: CurrentTopKPolicy(self.topk_capacity),
            POLICY_TOPK_ORDER: TopKOrderLRUPolicy(self.device_capacity),
            POLICY_HIT_MISS_ORDER: StableHitMissLRUPolicy(self.device_capacity),
            POLICY_AGE_PROBATION: AgeProbationLRUPolicy(
                self.device_capacity, self.probation_age
            ),
        }

    def process(self, record: TopKRecord) -> None:
        valid_topk = tuple(token for token in record.topk if token >= 0)
        if len(valid_topk) != len(set(valid_topk)):
            raise ValueError(
                "duplicate valid token in rank={} rid={} layer={} round={}".format(
                    record.rank,
                    record.request_key,
                    record.layer_id,
                    record.decode_round,
                )
            )
        if len(valid_topk) > self.topk_capacity:
            raise ValueError(
                "valid top-k length {} exceeds configured top-k capacity {}".format(
                    len(valid_topk), self.topk_capacity
                )
            )

        stream_key = record.stream_key
        previous = self._last_round.get(stream_key)
        if previous is not None and record.decode_round <= previous[0]:
            raise ValueError(
                "non-increasing decode round for rank={} rid={} layer={}: "
                "previous round={}, current round={}".format(
                    record.rank,
                    record.request_key,
                    record.layer_id,
                    previous[0],
                    record.decode_round,
                )
            )
        if previous is not None and record.seq_len <= previous[1]:
            raise ValueError(
                "non-increasing seq_len for rank={} rid={} layer={}: "
                "previous seq_len={}, current seq_len={}".format(
                    record.rank,
                    record.request_key,
                    record.layer_id,
                    previous[1],
                    record.seq_len,
                )
            )
        self._last_round[stream_key] = (record.decode_round, record.seq_len)

        policies = self._policies.get(stream_key)
        if policies is None:
            policies = self._new_policies()
            self._policies[stream_key] = policies

        layer_key = (record.rank, record.layer_id)
        layer_stats = self.by_layer.get(layer_key)
        if layer_stats is None:
            layer_stats = self._new_stats()
            self.by_layer[layer_key] = layer_stats

        request_key = (record.rank, record.request_key)
        request_stats = self.by_request.get(request_key)
        if request_stats is None:
            request_stats = self._new_stats()
            self.by_request[request_key] = request_stats

        request_layer_stats = self.by_request_layer.get(stream_key)
        if request_layer_stats is None:
            request_layer_stats = self._new_stats()
            self.by_request_layer[stream_key] = request_layer_stats

        for name in POLICY_NAMES:
            hits, misses = policies[name].access(valid_topk)
            self.overall[name].add(hits, misses)
            layer_stats[name].add(hits, misses)
            request_stats[name].add(hits, misses)
            request_layer_stats[name].add(hits, misses)


def _iter_lines(path: str) -> Iterator[Tuple[int, str]]:
    if path == "-":
        for line_number, line in enumerate(sys.stdin, start=1):
            yield line_number, line
        return

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            yield line_number, line


def analyze_logs(paths: Sequence[str], analyzer: HitRateAnalyzer) -> Tuple[int, int]:
    matched_lines = 0
    ignored_lines = 0
    stdin_count = sum(path == "-" for path in paths)
    if stdin_count > 1:
        raise ValueError("stdin ('-') may be specified only once")

    for path in paths:
        for line_number, line in _iter_lines(path):
            try:
                record = parse_log_line(line)
                if record is None:
                    ignored_lines += 1
                    continue
                analyzer.process(record)
                matched_lines += 1
            except (SyntaxError, ValueError) as exc:
                raise ValueError("{}:{}: {}".format(path, line_number, exc)) from exc
    return matched_lines, ignored_lines


def _print_stats_group(
    title: str,
    rows: Iterable[Tuple[str, Dict[str, HitRateStats]]],
    output: TextIO,
) -> None:
    print("\n{}".format(title), file=output)
    print(
        "{:<48} {:<28} {:>12} {:>12} {:>12} {:>12} {:>9}".format(
            "scope", "strategy", "hits", "misses", "accesses", "rounds", "hit_rate"
        ),
        file=output,
    )
    for scope, stats_by_policy in rows:
        for policy_name in POLICY_NAMES:
            stats = stats_by_policy[policy_name]
            print(
                "{:<48} {:<28} {:>12} {:>12} {:>12} {:>12} {:>8.4%}".format(
                    scope,
                    policy_name,
                    stats.hits,
                    stats.misses,
                    stats.accesses,
                    stats.rounds,
                    stats.hit_rate,
                ),
                file=output,
            )


def print_report(
    analyzer: HitRateAnalyzer,
    matched_lines: int,
    ignored_lines: int,
    details: bool,
    output: TextIO = sys.stdout,
) -> None:
    print(
        "parsed_records={} ignored_lines={} topk_capacity={} "
        "device_capacity={} probation_age={}".format(
            matched_lines,
            ignored_lines,
            analyzer.topk_capacity,
            analyzer.device_capacity,
            analyzer.probation_age,
        ),
        file=output,
    )
    _print_stats_group("Overall", [("all", analyzer.overall)], output)
    _print_stats_group(
        "Per layer",
        (
            ("rank={} layer={}".format(rank, layer_id), stats)
            for (rank, layer_id), stats in sorted(analyzer.by_layer.items())
        ),
        output,
    )
    if not details:
        return
    _print_stats_group(
        "Per request",
        (
            ("rank={} rid={}".format(rank, rid), stats)
            for (rank, rid), stats in sorted(analyzer.by_request.items())
        ),
        output,
    )
    _print_stats_group(
        "Per request and layer",
        (
            ("rank={} rid={} layer={}".format(rank, rid, layer_id), stats)
            for (rank, rid, layer_id), stats in sorted(
                analyzer.by_request_layer.items()
            )
        ),
        output,
    )


def write_csv_report(path: str, analyzer: HitRateAnalyzer) -> None:
    rows = []
    rows.append(("overall", "all", analyzer.overall))
    rows.extend(
        ("layer", "rank={} layer={}".format(*key), stats)
        for key, stats in sorted(analyzer.by_layer.items())
    )
    rows.extend(
        ("request", "rank={} rid={}".format(*key), stats)
        for key, stats in sorted(analyzer.by_request.items())
    )
    rows.extend(
        (
            "request_layer",
            "rank={} rid={} layer={}".format(*key),
            stats,
        )
        for key, stats in sorted(analyzer.by_request_layer.items())
    )
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "scope",
                "strategy",
                "hits",
                "misses",
                "accesses",
                "rounds",
                "hit_rate",
            ]
        )
        for group, scope, stats_by_policy in rows:
            for policy_name in POLICY_NAMES:
                stats = stats_by_policy[policy_name]
                writer.writerow(
                    [
                        group,
                        scope,
                        policy_name,
                        stats.hits,
                        stats.misses,
                        stats.accesses,
                        stats.rounds,
                        "{:.10f}".format(stats.hit_rate),
                    ]
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay SPARSE_KV_DECODE_TOPK logs on CPU and compare hit rates.",
        epilog=(
            "Strategy 1 replaces a 2048-entry cache with each round's top-k. "
            "Strategy 2 refreshes all top-k entries in top-k order. Strategy 3 "
            "uses stable cache-order hits followed by top-k-order misses. "
            "Strategy 4 reproduces fixed-slot timestamp ages with probation."
        ),
    )
    parser.add_argument(
        "logs", nargs="+", help="log file(s), .gz file(s), or '-' for stdin"
    )
    parser.add_argument(
        "--probation-age",
        type=int,
        default=4,
        help="initial age assigned to misses in strategy 4 (default: 4)",
    )
    parser.add_argument(
        "--topk-capacity",
        type=int,
        default=2048,
        help="strategy-1 capacity (default: 2048)",
    )
    parser.add_argument(
        "--device-capacity",
        type=int,
        default=4096,
        help="strategy-2/3/4 capacity (default: 4096)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="also print per-request and per-request-layer tables",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="write all aggregation levels to a CSV file",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        analyzer = HitRateAnalyzer(
            topk_capacity=args.topk_capacity,
            device_capacity=args.device_capacity,
            probation_age=args.probation_age,
        )
        matched_lines, ignored_lines = analyze_logs(args.logs, analyzer)
        if matched_lines == 0:
            raise ValueError("no SPARSE_KV_DECODE_TOPK records found")
        print_report(analyzer, matched_lines, ignored_lines, args.details)
        if args.csv:
            write_csv_report(args.csv, analyzer)
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
