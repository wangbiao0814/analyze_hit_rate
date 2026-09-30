#!/usr/bin/env bash
# 7280Z topology: 8 nodes, 80 logical CPUs/node, adjacent SMT siblings.
set -euo pipefail
if [[ $# -gt 1 || ${1:-} == --help || ${1:-} == -h ]]; then
  printf 'Usage: bash %s [benchmark_binary]\n' "$0"
  printf 'Environment: CORES_PER_NODE="8 16 24 32 40" CPU_MODE=physical|smt\n'
  printf 'KERNELS="native packed-sve"\n'
  printf 'SEQ_LEN=131072 HEADS=64 DIM=128 WARMUP=10 ITERS=100 TARGET_US=75\n'
  printf 'HUGE_PAGES=thp MEMORY_POLICY=bind QUERY_MODE=refresh DRY_RUN=0\n'
  printf 'CONTROLLER_CPU=N (optional; must be a spare physical core)\n'
  exit 0
fi
bench_bin=${1:-/tmp/numa_indexer_bench}
dry_run=${DRY_RUN:-0}
cpu_mode=${CPU_MODE:-physical}
[[ $dry_run == 0 || $dry_run == 1 ]] || { printf 'DRY_RUN must be 0 or 1\n' >&2; exit 2; }
[[ $cpu_mode == physical || $cpu_mode == smt ]] || { printf 'CPU_MODE must be physical or smt\n' >&2; exit 2; }
if [[ $dry_run == 0 && ! -x $bench_bin ]]; then
  printf 'Build the benchmark first: %s is not executable\n' "$bench_bin" >&2
  exit 1
fi
read -r -a core_counts <<< "${CORES_PER_NODE:-8 16 24 32 40}"
read -r -a kernels <<< "${KERNELS:-native packed-sve}"
for cores in "${core_counts[@]}"; do
  [[ $cores =~ ^([1-9]|[1-3][0-9]|40)$ ]] || { printf 'Invalid core count: %s\n' "$cores" >&2; exit 2; }
  cpus=""
  for ((node = 0; node < 8; ++node)); do
    for ((core = 0; core < cores; ++core)); do
      cpu=$((node * 80 + core * 2))
      cpus="${cpus:+$cpus,}$cpu"
      if [[ $cpu_mode == smt ]]; then
        cpus="$cpus,$((cpu + 1))"
      fi
    done
  done
  threads_per_node=$cores
  [[ $cpu_mode == smt ]] && threads_per_node=$((cores * 2))
  for kernel in "${kernels[@]}"; do
    args=("$bench_bin" --cpus "$cpus"
      --seq-len "${SEQ_LEN:-131072}" --heads "${HEADS:-64}" --dim "${DIM:-128}"
      --kernel "$kernel" --huge-pages "${HUGE_PAGES:-thp}"
      --memory-policy "${MEMORY_POLICY:-bind}" --query-mode "${QUERY_MODE:-refresh}"
      --warmup "${WARMUP:-10}" --iters "${ITERS:-100}" --target-us "${TARGET_US:-75}" --check)
    if [[ -n ${CONTROLLER_CPU:-} ]]; then
      args+=(--controller-cpu "$CONTROLLER_CPU")
    fi
    printf '\ncpu_mode=%s physical_cores_per_node=%d threads_per_node=%d total_threads=%d kernel=%s\n' \
      "$cpu_mode" "$cores" "$threads_per_node" "$((threads_per_node * 8))" "$kernel"
    if [[ $dry_run == 1 ]]; then
      printf '%q ' "${args[@]}"; printf '\n'
    else
      "${args[@]}"
    fi
  done
done
