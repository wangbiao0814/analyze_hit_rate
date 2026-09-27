import io
import os
import tempfile
import unittest

from analyze_sparse_kv_topk import (
    HitRateAnalyzer,
    POLICY_NAMES,
    ProgressReporter,
    analyze_one_request,
)


def record(
    rid: str,
    layer: int,
    decode_round: int,
    seq_len: int,
    topk: str = "[0, 1]",
    req_pool_idx: int = 3,
    rank: int = 0,
) -> str:
    return (
        "SPARSE_KV_DECODE_TOPK rank={} rid={} req_pool_idx={} "
        "layer_id={} decode_round={} seq_len={} topk={}\n"
    ).format(rank, rid, req_pool_idx, layer, decode_round, seq_len, topk)


class AnalyzeOneRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        self.addCleanup(lambda: os.unlink(handle.name))
        self.path = handle.name
        with handle:
            handle.write("server startup\n")
            handle.write(record("req-1", 0, 1, 10))
            handle.write(record("req-1", 1, 1, 10))
            handle.write(record("req-1", 0, 2, 11, "[1, 2]"))
            handle.write(record("req-1", 1, 2, 11, "[1, 3]"))
            handle.write("request finished\n")
            # An ordinary log line separates two contiguous top-k blocks even
            # when the server reuses the same rid and pool slot.
            handle.write(record("req-1", 0, 1, 20, "[4, 5]"))
            handle.write(record("req-1", 1, 1, 20, "[4, 6]"))
            handle.write(record("req-2", 0, 1, 30, "[7, 8]", 4))

    @staticmethod
    def analyzer() -> HitRateAnalyzer:
        return HitRateAnalyzer(topk_capacity=2, device_capacity=4)

    def test_ordinary_line_ends_reused_request_id_instance(self) -> None:
        analyzer = self.analyzer()
        matched, ignored, window = analyze_one_request(
            self.path, analyzer, start_line=1
        )

        self.assertEqual(matched, 4)
        self.assertEqual(ignored, 1)
        self.assertEqual(window.first_record_line, 2)
        self.assertEqual(window.end_line, 5)
        self.assertEqual(window.next_start_line, 6)
        self.assertEqual(set(analyzer.by_layer), {(0, 0), (0, 1)})
        for policy in POLICY_NAMES:
            self.assertEqual(analyzer.overall[policy].rounds, 4)

    def test_next_cursor_analyzes_only_the_next_instance(self) -> None:
        analyzer = self.analyzer()
        matched, ignored, window = analyze_one_request(
            self.path, analyzer, start_line=6
        )

        self.assertEqual(matched, 2)
        self.assertEqual(ignored, 1)
        self.assertEqual(window.first_record_line, 7)
        self.assertEqual(window.end_line, 8)
        self.assertEqual(window.next_start_line, 9)
        self.assertEqual(analyzer.overall[POLICY_NAMES[0]].rounds, 2)

    def test_rejects_non_positive_start_line(self) -> None:
        with self.assertRaisesRegex(ValueError, "start-line must be >= 1"):
            analyze_one_request(self.path, self.analyzer(), start_line=0)

    def test_rank_and_pool_changes_do_not_split_contiguous_block(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        self.addCleanup(lambda: os.unlink(handle.name))
        with handle:
            handle.write(record("req-x", 0, 1, 10, req_pool_idx=3, rank=0))
            handle.write(record("req-x", 1, 1, 10, req_pool_idx=8, rank=1))
            handle.write("ordinary separator\n")

        analyzer = self.analyzer()
        matched, ignored, window = analyze_one_request(
            handle.name, analyzer, start_line=1
        )

        self.assertEqual(matched, 2)
        self.assertEqual(ignored, 0)
        self.assertEqual(window.end_line, 2)
        self.assertEqual(window.next_start_line, 3)

    def test_progress_reports_current_record_and_final_counts(self) -> None:
        output = io.StringIO()
        progress = ProgressReporter(every=2, output=output)

        analyze_one_request(
            self.path,
            self.analyzer(),
            start_line=1,
            progress=progress,
        )

        text = output.getvalue()
        self.assertIn(
            "request=req-1 analyzed_layers=1 analyzed_decode_rounds=1 "
            "topk_records=1",
            text,
        )
        self.assertIn(
            "request=req-1 analyzed_layers=2 analyzed_decode_rounds=2 "
            "topk_records=4",
            text,
        )
        self.assertIn("line=5 scanned=5 records=4 ignored=1", text)


if __name__ == "__main__":
    unittest.main()
