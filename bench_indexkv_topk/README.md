## DeepSeek kernels benchmark


### Prerequisites
- You should install [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) from source before run `benchmark_deepgemm_fp8_gemm.py` and `benchmark_deepgemm_fp8_group_gemm.py`.

### Benchmark
- `benchmark_deepgemm_fp8_gemm.py`
    ```bash
    python benchmark_deepgemm_fp8_gemm.py --run_correctness --tp_size 1
    ```

- `benchmark_deepgemm_fp8_group_gemm.py`
    ```bash
    python benchmark_deepgemm_fp8_group_gemm.py --run_correctness --tp_size 1
    ```

 - You can use the `--run_correctness` parameter to verify all kernels results's correctness.
    - You can use the `--tp_size` parameter to benchmark all FP8 w8a8 block-wise matrix multiplications involved in DeepSeek V3/R1 under the current tensor parallelism (TP) setting. This benchmark compares DeepSeek's open-source [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) implementation with SGLang's and VLLM Triton implementation.

### CPU DSA indexer score (Arm SVE BF16)

`cpu_indexer_topk_bench.cpp` is a standalone DeepSeek-V3.2 indexer benchmark.
It computes BF16 index scores with FP32 accumulation using pthread workers. On Arm
it selects SVE BF16 `BFDOT`, then NEON BF16 `BFDOT`, with a portable scalar
fallback for build and correctness checks. Tokens are split into contiguous
`seq_len / threads` ranges; the final worker also handles any remainder.

Build on the target Arm server with native CPU features enabled:

```bash
g++ -O3 -std=c++17 -mcpu=native -pthread \
  bench_indexkv_topk/cpu_indexer_topk_bench.cpp \
  -o /tmp/cpu_indexer_score_bench
```

Run the DeepSeek-V3.2 128K configuration:

```bash
/tmp/cpu_indexer_score_bench \
  --seq-len 131072 --heads 64 --dim 128 \
  --threads 16 --warmup 3 --iters 20 --check
```

The first line should report `kernel=Arm SVE BF16 BFDOT`. If it reports the
scalar kernel, verify that the compiler supports SVE BF16 and that
`-mcpu=native` enables `__ARM_FEATURE_SVE_BF16`. The binary prints score timing,
effective throughput, memory bandwidth, and a checksum.

To isolate sequential index-K read bandwidth from the BF16 dot-product work,
use the lightweight memory-only kernel. It reads every index-K element exactly
once and uses a NEON XOR reduction only to keep the loads observable:

```bash
/tmp/cpu_indexer_score_bench \
  --seq-len 131072 --heads 64 --dim 128 \
  --memory-only --warmup 0 --iters 1 --check
```

For this mode `heads` is ignored because the unique index-K tensor has shape
`[seq_len, dim]`. A 128K × 128 BF16 tensor is only 32 MiB and may fit in the
last-level cache. Use `--warmup 0 --iters 1` for a first-pass measurement;
warmup iterations intentionally measure the cache-hot path instead.

### Kunpeng DRAM / L3 prefetch ceiling

`numa_dram_prefetch_bench.cpp` measures the sustainable read bandwidth that a
CPU-side indexer can use. Every pthread is pinned to a selected CPU and
first-touches its own buffer after pinning. With the normal Linux first-touch
NUMA policy, selecting CPUs from every NUMA node therefore creates local memory
traffic on every node instead of making all threads read memory allocated on
node 0.

Build it on the Kunpeng server:

```bash
g++ -O3 -std=c++17 -mcpu=native -pthread \
  bench_indexkv_topk/numa_dram_prefetch_bench.cpp \
  -o /tmp/numa_dram_prefetch_bench
```

First inspect the CPU-to-NUMA mapping, then select several physical cores from
each node (avoid SMT siblings if the machine exposes them):

```bash
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware

/tmp/numa_dram_prefetch_bench \
  --threads 32 \
  --cpus 0-7,32-39,64-71,96-103 \
  --buffer-mib 128 \
  --huge-pages thp \
  --prefetch-distance 0 \
  --warmup 1 --iters 10
```

The aggregate working set should be several times larger than the aggregate L3
cache. Sweep the software prefetch distance and compare against the zero-distance
hardware-prefetch baseline:

```bash
for distance in 0 256 512 1024 2048 4096; do
  /tmp/numa_dram_prefetch_bench \
    --threads 32 --cpus 0-7,32-39,64-71,96-103 \
    --buffer-mib 128 --prefetch-distance "$distance" \
    --warmup 1 --iters 10
done
```

The default `--huge-pages thp` mode calls `MADV_HUGEPAGE`. THP is only a hint,
so it does not prove that the mapping uses huge pages. For guaranteed explicit
HugeTLB mappings, reserve pages on every NUMA node first and select either 2 MiB
or 1 GiB pages:

```bash
grep -E 'HugePages|Hugepagesize|Hugetlb' /proc/meminfo

# Example only: reserve enough 2 MiB pages separately on each node.
echo 512 | sudo tee \
  /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages

/tmp/numa_dram_prefetch_bench \
  --threads 32 --cpus 0-7,32-39,64-71,96-103 \
  --buffer-mib 128 --huge-pages 2m \
  --prefetch-distance 512 --warmup 1 --iters 10
```

Repeat the reservation for every node used by `--cpus`. The total reserved
capacity on each node must cover the buffers first-touched by workers on that
node. `--huge-pages 1g` requires `--buffer-mib` to be a multiple of 1024 and
usually requires 1 GiB pages to have been reserved at boot. Explicit HugeTLB
allocation fails instead of silently falling back when the node's pool is too
small. Use `--huge-pages off` to force ordinary pages.

On AArch64, a non-zero distance emits `PRFM PLDL3KEEP`. This instruction is a
cache-placement hint, so the CPU is allowed to implement it differently; the
reported value is effective end-to-end streaming bandwidth rather than a pure
DRAM-to-L3 link measurement.

To judge whether the lightning indexer is memory-bound, compare this benchmark's
`aggregate_read_gib_per_s` with `cpu_indexer_topk_bench`'s
`unique_index_k_gib_per_s` at the same thread placement. If the indexer is well
below the streaming ceiling while CPU execution units are busy, it is not
limited by unique index-K DRAM bandwidth. A 128K × 128 BF16 index-K is only
32 MiB, so also compare a cold run (`--warmup 0 --iters 1`) with warmed runs.
