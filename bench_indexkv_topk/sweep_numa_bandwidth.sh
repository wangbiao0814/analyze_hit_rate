#!/usr/bin/env bash
# Target topology: 8 NUMA nodes, 80 logical CPUs per node, SMT siblings 2k/2k+1.
# Each sweep uses one logical CPU (the even ID) per selected physical core.
set -euo pipefail

if [[ $# -gt 1 || ${1:-} == --help || ${1:-} == -h ]]; then
  printf 'Usage: bash %s [benchmark_binary]\n' "$0"
  printf 'Environment: BUFFER_MIB=1024 HUGE_PAGES=thp PREFETCH_DISTANCE=0 WARMUP=3 ITERS=10 DRY_RUN=0\n'
  printf 'Fixed topology: nodes 0-7, node N CPUs [80*N, 80*N+79], adjacent SMT siblings.\n'
  exit 0
fi

bench_bin=${1:-/tmp/numa_dram_prefetch_bench}
buffer_mib=${BUFFER_MIB:-1024}
huge_pages=${HUGE_PAGES:-thp}
prefetch_distance=${PREFETCH_DISTANCE:-0}
warmup=${WARMUP:-3}
iters=${ITERS:-10}
dry_run=${DRY_RUN:-0}

if [[ $dry_run != 0 && $dry_run != 1 ]]; then
  printf 'error: DRY_RUN must be 0 or 1\n' >&2
  exit 2
fi
if [[ $dry_run == 0 && ! -x $bench_bin ]]; then
  printf 'error: benchmark is not executable: %s\nBuild it first using the README instructions.\n' "$bench_bin" >&2
  exit 1
fi

for cores_per_node in 8 16 24 32 40; do
  cpus=""
  for ((node = 0; node < 8; ++node)); do
    for ((core = 0; core < cores_per_node; ++core)); do
      cpu=$((node * 80 + core * 2))
      cpus="${cpus:+$cpus,}$cpu"
    done
  done

  command_args=(
    "$bench_bin"
    --threads "$((8 * cores_per_node))"
    --cpus "$cpus"
    --allocation-scope numa
    --buffer-mib "$buffer_mib"
    --huge-pages "$huge_pages"
    --prefetch-distance "$prefetch_distance"
    --warmup "$warmup" --iters "$iters"
  )

  printf '\ncores_per_node=%d total_threads=%d\n' "$cores_per_node" "$((8 * cores_per_node))"
  if [[ $dry_run == 1 ]]; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
  else
    "${command_args[@]}"
  fi
done
