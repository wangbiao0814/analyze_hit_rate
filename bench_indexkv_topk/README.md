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

### CPU DSA indexer top-k (Arm SVE BF16)

`cpu_indexer_topk_bench.cpp` is a standalone DeepSeek-V3.2 indexer benchmark.
It computes BF16 index scores with FP32 accumulation, followed by a parallel
top-k. On Arm it selects SVE BF16 `BFDOT`, then NEON BF16 `BFDOT`, with a
portable scalar fallback for build and correctness checks.

Build on the target Arm server with OpenMP and native CPU features enabled:

```bash
g++ -O3 -std=c++17 -mcpu=native -fopenmp \
  benchmark/kernels/deepseek/cpu_indexer_topk_bench.cpp \
  -o /tmp/cpu_indexer_topk_bench
```

Run the DeepSeek-V3.2 128K configuration:

```bash
OMP_PLACES=cores OMP_PROC_BIND=close \
  /tmp/cpu_indexer_topk_bench \
  --seq-len 131072 --heads 64 --dim 128 --topk 2048 \
  --threads 64 --warmup 3 --iters 20 --check
```

The first line should report `kernel=Arm SVE BF16 BFDOT`. If it reports the
scalar kernel, verify that the compiler supports SVE BF16 and that
`-mcpu=native` enables `__ARM_FEATURE_SVE_BF16`. The binary prints score,
top-k, and end-to-end timings separately. For NUMA systems, compare a
single-node run (`numactl --cpunodebind=0 --membind=0`) against a cross-node run
with `OMP_PROC_BIND=spread`; input initialization uses parallel first-touch.
