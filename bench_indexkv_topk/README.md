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

With SVE BF16 enabled, the first line should report `kernel=Arm SVE BF16 BFDOT`.
Feature detection is shared in `indexer_features.h`: it accepts
`__ARM_FEATURE_SVE_BF16`, or, for GCC (not Clang), the combination of
`__ARM_FEATURE_SVE` and `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`. Older GCC releases
can provide the intrinsics without the combined macro; see the
[GCC 15 release notes](https://gcc.gnu.org/gcc-15/changes.html).
`__ARM_FEATURE_SVE_BITS=0` is not a reason to disable SVE: the kernels use the
runtime vector length. If the binary still reports NEON or scalar, check the
actual compiler flags and executable path. The binary prints score timing,
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

Workers use no mutex or condition variable. An atomic generation number starts
each scan; each worker publishes completion to its own cache-line-aligned atomic
flag. The read loop contains no synchronization. Waiting threads spin with CPU
relax hints and periodically yield, so they still consume CPU time between scans.

Timing output distinguishes the following scopes:

- `stream_ms` and `aggregate_read_gib_per_s`: the earliest worker read start
  through the latest worker read end, including any stagger between workers.
- `numa_node_N_read_gib_per_s`: the same read-window definition, restricted to
  workers on node N. Separate node windows can overlap only partially, so their
  reported rates must not simply be added.
- `end_to_end_ms` and `aggregate_end_to_end_gib_per_s`: controller dispatch
  through observation of all completions, including synchronization overhead.
- `worker_start_skew_ms`: latest minus earliest worker read start in each scan.

Previously `aggregate_read_gib_per_s` included condition-variable wakeup and
completion overhead. Compare that older metric with the new
`aggregate_end_to_end_gib_per_s`, not directly with the new read-window metric.

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

### 7280Z NUMA 线程数扫描脚本

`sweep_numa_bandwidth.sh` 用于当前已确认的拓扑：8 个 NUMA 节点，每节点
80 个逻辑 CPU；node0 为 `0-79`，node1 为 `80-159`，依此类推；相邻编号
`0/1`、`2/3` 等属于同一物理核。脚本选择偶数编号，每个物理核只使用一个
SMT 线程，依次测试每节点 `8、16、24、32、40` 个物理核。

先按上面的命令编译 benchmark，然后在项目根目录运行：

```bash
bash bench_indexkv_topk/sweep_numa_bandwidth.sh

# 也可以传入自己编译的程序路径。
bash bench_indexkv_topk/sweep_numa_bandwidth.sh /path/to/numa_dram_prefetch_bench
```

默认每节点分配 1024 MiB，使用 THP，关闭软件预取，预热 3 轮、测量 10 轮。
完整保留每一组 benchmark 的输出，包括各节点和整机带宽。比较
`aggregate_read_gib_per_s` 的 `median_time`，找出带宽进入平台期的最小线程数。

可通过环境变量调整参数，或仅预览生成的 CPU 列表和命令：

```bash
BUFFER_MIB=1024 HUGE_PAGES=2m PREFETCH_DISTANCE=512 WARMUP=3 ITERS=20 \
  bash bench_indexkv_topk/sweep_numa_bandwidth.sh

DRY_RUN=1 bash bench_indexkv_topk/sweep_numa_bandwidth.sh
```

显式 HugeTLB 模式仍需要事先配置大页池。脚本使用固定拓扑；换机器时先核对
`lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE`，不能直接假设偶数 CPU 都对应不同物理核。

### NUMA index-K logits：128K / 75 μs 目标

`numa_indexer_bench.cpp` 将 NUMA 分片和 BF16 打分组合起来：

```text
logits[s] = sum_h weights[h] * max(dot(q[h, :], index_k[s, :]), 0)
```

`--seq-len 131072` 是全机的总 token 数。8 个节点均分时，每节点计算 16384 个
token，持有 4 MiB BF16 K（dim=128），全机 K 共 32 MiB。各节点分别映射 K、Q、
weights；默认使用 `mbind(MPOL_BIND)` 约束 K/Q/weights 的 NUMA 归属，绑核线程
初始化自己的 K 分片。所有线程直接写一个连续的 FP32 logits 数组，无需最后归并。
输出通过绑核线程 first-touch；小规模非整页分片的边界页可能由同节点或相邻节点
线程共享，默认 128K / 8 节点的输出分界是页对齐的。

线程常驻，读取循环无锁，轮次间使用原子 generation 和独立完成标记。节点之间
并行，节点内按输出 cache line 分片，处理不能整除的尾部。现有 benchmark 与组合版
共享 `indexer_bf16_kernels.h`，保留原生 SVE/NEON/portable 路径作为对照。

在目标 Kunpeng 上编译：

```bash
g++ -O3 -std=c++17 -mcpu=native -pthread \
  bench_indexkv_topk/numa_indexer_bench.cpp -o /tmp/numa_indexer_bench
```

每节点 8 个独立物理核的调用示例（按当前 7280Z 编号）：

```bash
/tmp/numa_indexer_bench \
  --cpus 0,2,4,6,8,10,12,14,80,82,84,86,88,90,92,94,160,162,164,166,168,170,172,174,240,242,244,246,248,250,252,254,320,322,324,326,328,330,332,334,400,402,404,406,408,410,412,414,480,482,484,486,488,490,492,494,560,562,564,566,568,570,572,574 \
  --seq-len 131072 --heads 64 --dim 128 \
  --kernel auto --memory-policy bind --huge-pages thp \
  --query-mode refresh --warmup 10 --iters 100 --target-us 75 --check
```

更方便的方式是扫描每节点 `8、16、24、32、40` 个物理核以及两种 kernel：

```bash
bash bench_indexkv_topk/sweep_numa_indexer.sh

# 只运行每节点 32 核，以及自动选择的 kernel。
CORES_PER_NODE=32 CONTROLLER_CPU=78 KERNELS=auto ITERS=200 \
  bash bench_indexkv_topk/sweep_numa_indexer.sh

# 仅查看命令；不分配内存、不运行 benchmark。
DRY_RUN=1 bash bench_indexkv_topk/sweep_numa_indexer.sh
```

这里每节点 32 核时，CPU 78 所属物理核没有工作线程，可留给调度线程。
不要把 `CONTROLLER_CPU=78` 用于每节点 40 核的配置，因为那时 CPU 78 已是工作线程。
默认扫描会显式测试 `packed-sve`；若编译器/CPU 不支持 SVE BF16，请用 `KERNELS=auto`。

建议先在目标机器运行数值自测（包含奇数 dim、head 尾部及不足 4 个 token）：

```bash
g++ -O3 -std=c++17 -mcpu=native \
  bench_indexkv_topk/test_indexer_kernels.cpp -o /tmp/test_indexer_kernels
/tmp/test_indexer_kernels
```

支持 SVE BF16 时应显示 `pass cases=200` 和 `packed_sve=executed`。
本地已验证 portable 数值路径、模拟两节点分片/同步及 SVE 交叉编译；模拟测试不验证
Linux NUMA 物理页归属，也不代表目标机器的 SVE 执行精度或 128K 性能。
本地 UBSan 检查通过；ASan 在当前 macOS 环境中初始化失败，因此未完成 ASan 检查。

旧 GCC 特性宏兼容性可在任何带 C++ 编译器的机器上回归检查：

```bash
bash bench_indexkv_topk/test_indexer_features.sh
```

该测试只执行预处理，覆盖旧 GCC 缺少组合宏、可变 SVE 长度、NEON-only、SVE-only
及 Clang 分支；不会把模拟特性宏用于生成可执行文件。
如果目标 CPU 已确认具有 `svebf16` 和 `bf16`，可用
`-march=armv8.2-a+sve+bf16` 替代上述 `-mcpu=native`，重新编译数值自测和 benchmark。
不要手动定义编译器保留的 `__ARM_*` 宏。修复后 `native` 应显示
`Arm SVE BF16 BFDOT`，`packed-sve` 应显示 `packed SVE BF16 4-token`。

参数和测量口径：

- `--kernel native`：原有逐 token 的 BF16 kernel。
- `--kernel packed-sve`：将 Q 按 dimension-pair/head 打包，让 SVE 的 FP32 lane
  分别累加不同 head；同时处理 4 个 token，复用 Q 加载，最后进行 ReLU、权重求和。
  要求编译器启用 SVE BF16；不足 4 个 token 的尾部使用 native kernel。
  `auto` 在支持 SVE BF16 时选它，否则用 native。性能优劣需要在目标机器比较。
- `--query-mode refresh`（默认）：每轮交替使用两份预生成的不同 Q/weights，
  节点首线程复制本轮输入到本地并打包 Q，再启动本节点计算。复制、打包、等待都计入
  `end_to_end_us`。Q 的上游生成过程不计时。
- `--query-mode static`：Q/weights 和打包结果在初始化时就绪，用于隔离计算性能；
  不代表每轮新 Q 的实际调用成本。
- `kernel_us`：全局最早 kernel 开始到最晚 kernel 结束，包含节点启动错开。
  `end_to_end_us`：主线程发布本轮请求到确认所有连续 logits 可读取。
  `worker_start_skew_us`：最晚与最早 kernel 启动时间差。三者均输出 min/median/p95/max。
- `target_end_to_end_p95=pass`：端到端 p95 不超过 `--target-us`；同时输出达标轮数。
  这不是每一轮都不超过目标的保证，严格上界还要看 max。
- `--check`：用独立 FP64 reference 校验每线程首/中/尾 token 和全局均匀采样。
  `--check-all` 检查所有 token。校验在计时结束后进行；所有输出始终检查 finite。
- `--huge-pages` 作用于 K 映射，支持 off/thp/2m/1g。显式大页会向上对齐映射长度，
  实际计算和带宽只按真实 token 数统计；`k_mapping_mib` 显示分配容量。
- `--memory-policy first-touch` 可用于没有 mbind 权限的环境，依赖当前 Linux 内存
  策略，不能保证严格节点本地性。默认 bind 失败会报错，不自动降级。
- 可用 `--controller-cpu N` 将调度线程绑定到一个预留物理核；不要选择工作线程的
  SMT 兄弟核。等待线程会轮询并周期性 yield，因此线程数需要实测调优。

75 μs 是性能目标，并非已在本机验证的结果。默认形状点积部分约为 2.147 GFLOP，
75 μs 需要约 28.63 TFLOP/s；还不含 ReLU 和权重求和开销。K 初始化、内存分配、
线程创建和校验均不计入单轮耗时。K 在各轮复用，结果属于 resident/cache-hot 倾向
的 indexer 测试，不能把 `unique_k_gib_per_s` 当作实测 DRAM 流量。程序只计算 logits，
不包含 top-k、上游 Q 投影或与设备间的数据传输。

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
