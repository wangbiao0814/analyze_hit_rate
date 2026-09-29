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
CPU-side indexer can use. Every pthread is pinned to a selected CPU. The default
`--allocation-scope thread` gives each worker a private buffer. With
`--allocation-scope numa`, the program discovers the node of each selected CPU,
allocates one buffer per node, and has that node's workers first-touch and read
disjoint slices of it. Both modes create node-local memory traffic under the
normal Linux first-touch policy.

Each worker samples one `uint64_t` every eight elements in its slice
(`0, 8, 16, ...`, a 64-byte stride), adding the values to a checksum. The load
is volatile to preserve individual 8-byte reads. The `*_read_gib_per_s` metrics
use the scanned address span; `*_sampled_gib_per_s` counts only the loaded
8-byte values and is one eighth of the span bandwidth. Neither metric is a
hardware measurement of DRAM traffic; cache residency and prefetching affect
the actual traffic.

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
  --allocation-scope numa \
  --buffer-mib 1024 \
  --huge-pages thp \
  --prefetch-distance 0 \
  --warmup 1 --iters 10
```

In NUMA allocation mode, `--buffer-mib` is the size of each node's buffer, not
each worker's slice. The program reports both aggregate bandwidth and a bandwidth
line for every detected node. The aggregate working set should be several times
larger than the aggregate L3 cache. Sweep the software prefetch distance and
compare against the zero-distance hardware-prefetch baseline:

```bash
for distance in 0 256 512 1024 2048 4096; do
  /tmp/numa_dram_prefetch_bench \
    --threads 32 --cpus 0-7,32-39,64-71,96-103 \
    --allocation-scope numa --buffer-mib 1024 \
    --prefetch-distance "$distance" \
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
  --allocation-scope numa --buffer-mib 1024 --huge-pages 2m \
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

### CPU / DRAM 硬件查询

在 Kunpeng 服务器上执行以下只读命令，保留输出，用于核对 CPU 内存通道规格、
DIMM 插法和 NUMA 拓扑，估算每个 NUMA 节点、每颗 CPU 及整机的理论内存带宽。

#### CPU 型号、核数和 NUMA 拓扑

```bash
LC_ALL=C lscpu
LC_ALL=C lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware
```

#### CPU 完整型号和服务器型号

```bash
sudo dmidecode -t processor | grep -E \
'Socket Designation:|Manufacturer:|Version:|Current Speed:|Max Speed:|Core Count:|Thread Count:'

sudo dmidecode -s system-manufacturer
sudo dmidecode -s system-product-name
```

#### 内存槽位、容量和配置速率

```bash
sudo dmidecode -t 17 | grep -E \
'Memory Device$|Size:|Locator:|Type:|Speed:|Data Width:|Total Width:|Rank:|Manufacturer:|Part Number:'
```

保留所有槽位，包括 `No Module Installed`。重点关注：

- `Configured Memory Speed`：固件报告的配置速率，可能低于标称 `Speed`。
- `Locator` / `Bank Locator`：用于核对内存条与 CPU、通道的对应关系。
- `Data Width`：有效数据位宽；`Total Width` 可能包含 ECC 校验位。
- `Size`、`Rank` 和 `Part Number`：用于确认实际安装配置。

内存条数量不一定等于通道数量。需要结合具体 CPU 和服务器手册确认启用通道数，
以及这些通道与 NUMA 节点的对应关系，不能只用 DIMM 数或 NUMA 数直接推算。

#### L3 容量及共享 CPU 范围

```bash
LC_ALL=C lscpu -C

for cache in /sys/devices/system/cpu/cpu*/cache/index*; do
  [ -r "$cache/level" ] || continue
  [ "$(cat "$cache/level")" = 3 ] || continue
  printf 'L3 size=%s shared_cpus=%s\n' \
    "$(cat "$cache/size")" \
    "$(cat "$cache/shared_cpu_list")"
done | sort -u
```

旧版 `lscpu` 如果不支持 `-C`，使用后面的 sysfs 查询循环即可。
结合这些结果检查每节点测试工作集是否可能驻留在 L3，再比较实测带宽。
本程序输出的带宽单位为 GiB/s，换算为 GB/s 时乘以 `1.073741824`。

工具说明：[lscpu](https://man7.org/linux/man-pages/man1/lscpu.1.html)、
[dmidecode](https://www.nongnu.org/dmidecode/)。`dmidecode` 的信息由固件提供；
如果字段缺失或不明确，保留原始输出并结合服务器手册核实。
