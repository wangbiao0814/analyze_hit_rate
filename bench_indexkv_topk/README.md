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
It computes BF16 index scores with FP32 accumulation on one CPU thread. On Arm
it selects SVE BF16 `BFDOT`, then NEON BF16 `BFDOT`, with a portable scalar
fallback for build and correctness checks. The benchmark reports the single-core
score kernel throughput in GFLOPS.

Build on the target Arm server with native CPU features enabled:

```bash
g++ -O3 -std=c++17 -mcpu=native \
  bench_indexkv_topk/cpu_indexer_topk_bench.cpp \
  -o /tmp/cpu_indexer_score_bench
```

Run the DeepSeek-V3.2 128K configuration:

```bash
/tmp/cpu_indexer_score_bench \
  --seq-len 131072 --heads 64 --dim 128 \
  --warmup 3 --iters 20 --check
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
