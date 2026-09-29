// Linux/AArch64 streaming-read benchmark for estimating the DRAM bandwidth
// available to a CPU-side indexer. Each pthread is pinned to one CPU. Buffers
// can be allocated per worker or per NUMA node; in NUMA mode the workers on a
// node first-touch and stream disjoint slices of that node's buffer.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#if !defined(__linux__)
#error "numa_dram_prefetch_bench requires Linux"
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kElementStride = 8;
constexpr std::size_t kReadStrideBytes = kElementStride * sizeof(std::uint64_t);

enum class HugePageMode {
  kOff,
  kThp,
  kHugeTlb2M,
  kHugeTlb1G,
};

enum class AllocationScope {
  kThread,
  kNuma,
};

struct Options {
  int threads = 1;
  std::size_t buffer_mib = 256;
  std::size_t prefetch_distance = 0;
  int warmup = 1;
  int iters = 5;
  std::string cpu_list;
  HugePageMode huge_pages = HugePageMode::kThp;
  AllocationScope allocation_scope = AllocationScope::kThread;
};

struct TimingStats {
  double min_ms;
  double median_ms;
  double mean_ms;
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --threads N              worker count (default: 1)\n"
      << "  --cpus LIST              pinned CPUs, e.g. 0-7,32-39\n"
      << "  --buffer-mib N           MiB per allocation (default: 256)\n"
      << "  --allocation-scope MODE  thread or numa (default: thread)\n"
      << "  --prefetch-distance N    bytes ahead for AArch64 PLDL3KEEP; 0 disables\n"
      << "  --huge-pages MODE        off, thp, 2m, or 1g (default: thp)\n"
      << "  --warmup N               untimed scans (default: 1)\n"
      << "  --iters N                measured scans (default: 5)\n"
      << "  --no-thp                 alias for --huge-pages off\n"
      << "  --help                   show this message\n";
  std::exit(error.empty() ? 0 : 2);
}

template <typename T>
T parse_integer(const char* text, const char* name, bool allow_zero = false) {
  std::size_t consumed = 0;
  const std::string value(text);
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &consumed);
  } catch (const std::exception&) {
    usage("numa_dram_prefetch_bench", std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || (!allow_zero && parsed == 0) ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
    usage("numa_dram_prefetch_bench", std::string("invalid ") + name + ": " + value);
  }
  return static_cast<T>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> const char* {
      if (++i >= argc) {
        usage(argv[0], std::string("missing value for ") + name);
      }
      return argv[i];
    };

    if (arg == "--threads") {
      options.threads = parse_integer<int>(require_value("--threads"), "threads");
    } else if (arg == "--cpus") {
      options.cpu_list = require_value("--cpus");
    } else if (arg == "--buffer-mib") {
      options.buffer_mib =
          parse_integer<std::size_t>(require_value("--buffer-mib"), "buffer-mib");
    } else if (arg == "--allocation-scope") {
      const std::string scope = require_value("--allocation-scope");
      if (scope == "thread") {
        options.allocation_scope = AllocationScope::kThread;
      } else if (scope == "numa") {
        options.allocation_scope = AllocationScope::kNuma;
      } else {
        usage(argv[0], "allocation-scope must be thread or numa");
      }
    } else if (arg == "--prefetch-distance") {
      options.prefetch_distance = parse_integer<std::size_t>(
          require_value("--prefetch-distance"), "prefetch-distance", true);
    } else if (arg == "--huge-pages") {
      const std::string mode = require_value("--huge-pages");
      if (mode == "off") {
        options.huge_pages = HugePageMode::kOff;
      } else if (mode == "thp") {
        options.huge_pages = HugePageMode::kThp;
      } else if (mode == "2m") {
        options.huge_pages = HugePageMode::kHugeTlb2M;
      } else if (mode == "1g") {
        options.huge_pages = HugePageMode::kHugeTlb1G;
      } else {
        usage(argv[0], "huge-pages must be off, thp, 2m, or 1g");
      }
    } else if (arg == "--warmup") {
      options.warmup =
          parse_integer<int>(require_value("--warmup"), "warmup", true);
    } else if (arg == "--iters") {
      options.iters = parse_integer<int>(require_value("--iters"), "iters");
    } else if (arg == "--no-thp") {
      options.huge_pages = HugePageMode::kOff;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0], "unknown option: " + arg);
    }
  }

  if (options.buffer_mib >
      std::numeric_limits<std::size_t>::max() / (1024ULL * 1024ULL)) {
    usage(argv[0], "buffer-mib is too large");
  }
  if (options.prefetch_distance % 64 != 0) {
    usage(argv[0], "prefetch-distance must be a multiple of 64 bytes");
  }
  if (options.huge_pages == HugePageMode::kHugeTlb2M &&
      options.buffer_mib % 2 != 0) {
    usage(argv[0], "buffer-mib must be a multiple of 2 for 2 MiB HugeTLB pages");
  }
  if (options.huge_pages == HugePageMode::kHugeTlb1G &&
      options.buffer_mib % 1024 != 0) {
    usage(argv[0], "buffer-mib must be a multiple of 1024 for 1 GiB HugeTLB pages");
  }
  return options;
}

const char* huge_page_mode_name(HugePageMode mode) {
  switch (mode) {
    case HugePageMode::kOff:
      return "off";
    case HugePageMode::kThp:
      return "thp";
    case HugePageMode::kHugeTlb2M:
      return "hugetlb-2m";
    case HugePageMode::kHugeTlb1G:
      return "hugetlb-1g";
  }
  return "unknown";
}

const char* allocation_scope_name(AllocationScope scope) {
  return scope == AllocationScope::kNuma ? "numa" : "thread";
}

int parse_cpu_number(const std::string& text) {
  if (text.empty()) {
    throw std::runtime_error("empty CPU number in --cpus");
  }
  std::size_t consumed = 0;
  const long value = std::stol(text, &consumed);
  if (consumed != text.size() || value < 0 || value >= CPU_SETSIZE) {
    throw std::runtime_error("invalid CPU in --cpus: " + text);
  }
  return static_cast<int>(value);
}

std::vector<int> parse_cpu_list(const std::string& text) {
  std::vector<int> cpus;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::string item = text.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin);
    const std::size_t dash = item.find('-');
    if (dash == std::string::npos) {
      cpus.push_back(parse_cpu_number(item));
    } else {
      const int first = parse_cpu_number(item.substr(0, dash));
      const int last = parse_cpu_number(item.substr(dash + 1));
      if (last < first) {
        throw std::runtime_error("descending CPU range in --cpus: " + item);
      }
      for (int cpu = first; cpu <= last; ++cpu) {
        cpus.push_back(cpu);
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (cpus.empty()) {
    throw std::runtime_error("--cpus cannot be empty");
  }
  std::vector<int> unique = cpus;
  std::sort(unique.begin(), unique.end());
  if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
    throw std::runtime_error("--cpus contains a duplicate CPU");
  }
  return cpus;
}

std::vector<int> allowed_cpus() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    throw std::runtime_error(
        std::string("sched_getaffinity failed: ") + std::strerror(errno));
  }
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

int numa_node_for_cpu(int cpu) {
  // Linux exposes node membership as cpuN/nodeM symlinks. A system without
  // the NUMA sysfs hierarchy is treated as a single-node machine.
  if (access("/sys/devices/system/node", F_OK) != 0) {
    return 0;
  }
  for (int node = 0; node < CPU_SETSIZE; ++node) {
    const std::string path = "/sys/devices/system/cpu/cpu" +
                             std::to_string(cpu) + "/node" +
                             std::to_string(node);
    if (access(path.c_str(), F_OK) == 0) {
      return node;
    }
  }
  throw std::runtime_error(
      "cannot determine NUMA node for CPU " + std::to_string(cpu));
}

void prefetch_l3(const void* address) {
#if defined(__aarch64__)
  asm volatile("prfm pldl3keep, [%0]" : : "r"(address));
#else
  __builtin_prefetch(address, 0, 0);
#endif
}

__attribute__((noinline)) std::uint64_t stream_read(
    const std::uint8_t* data,
    std::size_t bytes,
    std::size_t prefetch_distance) {
  // Worker slices start at 64-byte boundaries and contain whole cache lines.
  // One volatile uint64_t load every eight elements prevents widened loads or
  // elimination of the sampled reads. Unsigned accumulation wraps modulo 2^64.
  std::uint64_t checksum = 0;
  for (std::size_t offset = 0; offset + sizeof(std::uint64_t) <= bytes;
       offset += kReadStrideBytes) {
    if (prefetch_distance != 0 && prefetch_distance < bytes - offset) {
      prefetch_l3(data + offset + prefetch_distance);
    }
    checksum += *reinterpret_cast<const volatile std::uint64_t*>(data + offset);
  }
  return checksum;
}

struct SharedState {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  int allocations_ready = 0;
  int allocation_count = 0;
  int ready = 0;
  int finished = 0;
  std::uint64_t generation = 0;
  bool stop = false;
};

struct BufferState {
  std::uint8_t* buffer = nullptr;
  std::size_t bytes = 0;
  int node = -1;
  int worker_count = 0;
  int allocation_error = 0;
  HugePageMode huge_pages = HugePageMode::kOff;
};

struct alignas(64) WorkerState {
  SharedState* shared = nullptr;
  BufferState* allocation = nullptr;
  std::uint8_t* buffer = nullptr;
  std::size_t bytes = 0;
  std::size_t prefetch_distance = 0;
  std::size_t local_rank = 0;
  int cpu = -1;
  int node = -1;
  int affinity_error = 0;
  std::uint64_t checksum = 0;
  Clock::time_point read_begin;
  Clock::time_point read_end;
};

int huge_page_mmap_flags(HugePageMode mode) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (mode == HugePageMode::kHugeTlb2M ||
      mode == HugePageMode::kHugeTlb1G) {
    flags |= MAP_HUGETLB;
    // Linux encodes log2(page_size) in the six bits starting at bit 26.
    constexpr int kMapHugeShift = 26;
    const int page_shift = mode == HugePageMode::kHugeTlb2M ? 21 : 30;
    flags |= page_shift << kMapHugeShift;
  }
  return flags;
}

void* worker_main(void* opaque) {
  WorkerState& worker = *static_cast<WorkerState*>(opaque);
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(worker.cpu, &affinity);
  worker.affinity_error = pthread_setaffinity_np(
      pthread_self(), sizeof(affinity), &affinity);

  BufferState& allocation = *worker.allocation;
  if (worker.local_rank == 0 && worker.affinity_error == 0) {
    void* mapping = mmap(nullptr,
                         allocation.bytes,
                         PROT_READ | PROT_WRITE,
                         huge_page_mmap_flags(allocation.huge_pages),
                         -1,
                         0);
    if (mapping == MAP_FAILED) {
      allocation.allocation_error = errno;
    } else {
      allocation.buffer = static_cast<std::uint8_t*>(mapping);
      if (allocation.huge_pages == HugePageMode::kThp &&
          madvise(mapping, allocation.bytes, MADV_HUGEPAGE) != 0) {
        allocation.allocation_error = errno;
      }
    }
  }

  SharedState& shared = *worker.shared;
  pthread_mutex_lock(&shared.mutex);
  if (worker.local_rank == 0) {
    ++shared.allocations_ready;
    pthread_cond_broadcast(&shared.condition);
  }
  while (!shared.stop &&
         shared.allocations_ready != shared.allocation_count) {
    pthread_cond_wait(&shared.condition, &shared.mutex);
  }
  if (shared.stop) {
    pthread_mutex_unlock(&shared.mutex);
    return nullptr;
  }
  pthread_mutex_unlock(&shared.mutex);

  if (worker.affinity_error == 0 && allocation.allocation_error == 0 &&
      allocation.buffer != nullptr) {
    const std::size_t cache_lines = allocation.bytes / 64;
    const std::size_t begin_line =
        cache_lines * worker.local_rank / allocation.worker_count;
    const std::size_t end_line =
        cache_lines * (worker.local_rank + 1) / allocation.worker_count;
    worker.buffer = allocation.buffer + begin_line * 64;
    worker.bytes = (end_line - begin_line) * 64;

    // Every worker first-touches its own slice after binding. In NUMA scope all
    // workers sharing this mapping are on the same node.
    std::memset(worker.buffer, worker.cpu + 1, worker.bytes);
  }

  pthread_mutex_lock(&shared.mutex);
  ++shared.ready;
  pthread_cond_broadcast(&shared.condition);
  std::uint64_t observed_generation = shared.generation;
  while (true) {
    while (!shared.stop && shared.generation == observed_generation) {
      pthread_cond_wait(&shared.condition, &shared.mutex);
    }
    if (shared.stop) {
      pthread_mutex_unlock(&shared.mutex);
      return nullptr;
    }
    observed_generation = shared.generation;
    pthread_mutex_unlock(&shared.mutex);

    worker.read_begin = Clock::now();
    worker.checksum += stream_read(
                           worker.buffer,
                           worker.bytes,
                           worker.prefetch_distance) +
                       observed_generation;
    worker.read_end = Clock::now();

    pthread_mutex_lock(&shared.mutex);
    ++shared.finished;
    pthread_cond_broadcast(&shared.condition);
  }
}

TimingStats summarize(std::vector<double> samples) {
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
                            ? (samples[middle - 1] + samples[middle]) * 0.5
                            : samples[middle];
  return {samples.front(), median, mean};
}

std::string join_cpus(const std::vector<int>& cpus) {
  std::string result;
  for (std::size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0) {
      result += ',';
    }
    result += std::to_string(cpus[i]);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<int> cpus = options.cpu_list.empty()
                                ? allowed_cpus()
                                : parse_cpu_list(options.cpu_list);
    if (static_cast<std::size_t>(options.threads) > cpus.size()) {
      throw std::runtime_error(
          "threads exceeds the number of CPUs supplied/allowed");
    }
    cpus.resize(static_cast<std::size_t>(options.threads));

    std::vector<int> cpu_nodes;
    cpu_nodes.reserve(cpus.size());
    for (int cpu : cpus) {
      cpu_nodes.push_back(numa_node_for_cpu(cpu));
    }

    std::vector<int> allocation_nodes;
    std::vector<std::size_t> allocation_for_worker(cpus.size());
    if (options.allocation_scope == AllocationScope::kThread) {
      allocation_nodes = cpu_nodes;
      for (std::size_t worker = 0; worker < cpus.size(); ++worker) {
        allocation_for_worker[worker] = worker;
      }
    } else {
      for (std::size_t worker = 0; worker < cpus.size(); ++worker) {
        auto found = std::find(
            allocation_nodes.begin(), allocation_nodes.end(), cpu_nodes[worker]);
        if (found == allocation_nodes.end()) {
          allocation_nodes.push_back(cpu_nodes[worker]);
          allocation_for_worker[worker] = allocation_nodes.size() - 1;
        } else {
          allocation_for_worker[worker] = static_cast<std::size_t>(
              found - allocation_nodes.begin());
        }
      }
    }

    const std::size_t bytes_per_allocation =
        options.buffer_mib * 1024ULL * 1024ULL;
    if (bytes_per_allocation >
        std::numeric_limits<std::size_t>::max() /
            allocation_nodes.size()) {
      throw std::runtime_error("aggregate working set is too large");
    }
    const std::size_t aggregate_bytes =
        bytes_per_allocation * allocation_nodes.size();

    SharedState shared;
    if (pthread_mutex_init(&shared.mutex, nullptr) != 0 ||
        pthread_cond_init(&shared.condition, nullptr) != 0) {
      throw std::runtime_error("failed to initialize pthread synchronization");
    }

    std::vector<BufferState> allocations(allocation_nodes.size());
    for (std::size_t allocation = 0; allocation < allocations.size(); ++allocation) {
      allocations[allocation].bytes = bytes_per_allocation;
      allocations[allocation].node = allocation_nodes[allocation];
      allocations[allocation].huge_pages = options.huge_pages;
    }
    for (std::size_t allocation : allocation_for_worker) {
      ++allocations[allocation].worker_count;
    }
    shared.allocation_count = static_cast<int>(allocations.size());

    std::vector<int> numa_nodes;
    for (int node : cpu_nodes) {
      if (std::find(numa_nodes.begin(), numa_nodes.end(), node) ==
          numa_nodes.end()) {
        numa_nodes.push_back(node);
      }
    }
    std::vector<std::size_t> bytes_per_node(numa_nodes.size(), 0);
    for (const BufferState& allocation : allocations) {
      const auto found =
          std::find(numa_nodes.begin(), numa_nodes.end(), allocation.node);
      bytes_per_node[static_cast<std::size_t>(found - numa_nodes.begin())] +=
          allocation.bytes;
    }

    std::vector<WorkerState> states(static_cast<std::size_t>(options.threads));
    std::vector<pthread_t> workers(static_cast<std::size_t>(options.threads));
    std::vector<std::size_t> next_local_rank(allocations.size(), 0);
    for (int worker = 0; worker < options.threads; ++worker) {
      const std::size_t allocation = allocation_for_worker[worker];
      states[worker].shared = &shared;
      states[worker].allocation = &allocations[allocation];
      states[worker].prefetch_distance = options.prefetch_distance;
      states[worker].local_rank = next_local_rank[allocation]++;
      states[worker].cpu = cpus[worker];
      states[worker].node = cpu_nodes[worker];
    }

    std::size_t created = 0;
    for (; created < workers.size(); ++created) {
      const int error = pthread_create(
          &workers[created], nullptr, worker_main, &states[created]);
      if (error != 0) {
        pthread_mutex_lock(&shared.mutex);
        shared.stop = true;
        pthread_cond_broadcast(&shared.condition);
        pthread_mutex_unlock(&shared.mutex);
        for (std::size_t worker = 0; worker < created; ++worker) {
          pthread_join(workers[worker], nullptr);
        }
        throw std::runtime_error(
            std::string("pthread_create failed: ") + std::strerror(error));
      }
    }

    bool workers_stopped = false;
    auto stop_workers = [&]() {
      if (workers_stopped) {
        return;
      }
      pthread_mutex_lock(&shared.mutex);
      shared.stop = true;
      pthread_cond_broadcast(&shared.condition);
      pthread_mutex_unlock(&shared.mutex);
      for (pthread_t worker : workers) {
        pthread_join(worker, nullptr);
      }
      workers_stopped = true;
    };

    auto release_buffers = [&]() {
      for (const BufferState& allocation : allocations) {
        if (allocation.buffer != nullptr) {
          munmap(allocation.buffer, allocation.bytes);
        }
      }
    };

    pthread_mutex_lock(&shared.mutex);
    while (shared.ready != options.threads) {
      pthread_cond_wait(&shared.condition, &shared.mutex);
    }
    pthread_mutex_unlock(&shared.mutex);

    std::string initialization_error;
    for (const WorkerState& worker : states) {
      if (worker.affinity_error != 0) {
        initialization_error =
            "failed to pin a worker to CPU " + std::to_string(worker.cpu) +
            ": " + std::strerror(worker.affinity_error);
        break;
      }
    }
    if (initialization_error.empty()) {
      for (const BufferState& allocation : allocations) {
        if (allocation.allocation_error != 0) {
          initialization_error =
              "memory allocation failed on NUMA node " +
              std::to_string(allocation.node) + " with huge-pages=" +
              huge_page_mode_name(allocation.huge_pages) + ": " +
              std::strerror(allocation.allocation_error);
          break;
        }
      }
    }
    if (!initialization_error.empty()) {
      stop_workers();
      release_buffers();
      pthread_cond_destroy(&shared.condition);
      pthread_mutex_destroy(&shared.mutex);
      throw std::runtime_error(initialization_error);
    }

    std::cout << "workers=" << options.threads << "\n"
              << "cpus=" << join_cpus(cpus) << "\n"
              << "allocation_scope="
              << allocation_scope_name(options.allocation_scope) << "\n"
              << "buffer_mib_per_allocation=" << options.buffer_mib << "\n"
              << "allocation_count=" << allocations.size() << "\n"
              << "element_bytes=" << sizeof(std::uint64_t) << "\n"
              << "read_stride_elements=" << kElementStride << "\n"
              << "read_bandwidth_basis=address_span (not measured DRAM traffic)\n"
              << "aggregate_working_set_gib=" << std::fixed
              << std::setprecision(3)
              << aggregate_bytes / static_cast<double>(1ULL << 30) << "\n"
              << "prefetch="
              << (options.prefetch_distance == 0 ? "disabled" : "AArch64 PLDL3KEEP")
              << "\n"
              << "prefetch_distance_bytes=" << options.prefetch_distance << "\n"
              << "huge_pages=" << huge_page_mode_name(options.huge_pages) << "\n";
    for (std::size_t node_index = 0; node_index < numa_nodes.size(); ++node_index) {
      std::vector<int> node_cpus;
      for (std::size_t worker = 0; worker < cpus.size(); ++worker) {
        if (cpu_nodes[worker] == numa_nodes[node_index]) {
          node_cpus.push_back(cpus[worker]);
        }
      }
      std::cout << "numa_node_" << numa_nodes[node_index]
                << "_cpus=" << join_cpus(node_cpus) << "\n"
                << "numa_node_" << numa_nodes[node_index]
                << "_working_set_gib="
                << bytes_per_node[node_index] /
                       static_cast<double>(1ULL << 30)
                << "\n";
    }

    auto run_once = [&]() {
      pthread_mutex_lock(&shared.mutex);
      shared.finished = 0;
      ++shared.generation;
      const auto begin = Clock::now();
      pthread_cond_broadcast(&shared.condition);
      while (shared.finished != options.threads) {
        pthread_cond_wait(&shared.condition, &shared.mutex);
      }
      const auto end = Clock::now();
      pthread_mutex_unlock(&shared.mutex);
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    for (int i = 0; i < options.warmup; ++i) {
      (void)run_once();
    }
    std::vector<double> samples;
    samples.reserve(options.iters);
    std::vector<std::vector<double>> node_samples(numa_nodes.size());
    for (std::vector<double>& node_sample : node_samples) {
      node_sample.reserve(options.iters);
    }
    for (int i = 0; i < options.iters; ++i) {
      samples.push_back(run_once());
      for (std::size_t node_index = 0; node_index < numa_nodes.size();
           ++node_index) {
        bool first_worker = true;
        Clock::time_point node_begin;
        Clock::time_point node_end;
        for (const WorkerState& worker : states) {
          if (worker.node != numa_nodes[node_index]) {
            continue;
          }
          if (first_worker || worker.read_begin < node_begin) {
            node_begin = worker.read_begin;
          }
          if (first_worker || worker.read_end > node_end) {
            node_end = worker.read_end;
          }
          first_worker = false;
        }
        node_samples[node_index].push_back(
            std::chrono::duration<double, std::milli>(node_end - node_begin)
                .count());
      }
    }

    stop_workers();

    std::uint64_t checksum = 0;
    for (std::size_t worker = 0; worker < states.size(); ++worker) {
      checksum += states[worker].checksum * (worker + 1);
    }
    release_buffers();
    pthread_cond_destroy(&shared.condition);
    pthread_mutex_destroy(&shared.mutex);

    const TimingStats stats = summarize(samples);
    const double gib = aggregate_bytes / static_cast<double>(1ULL << 30);
    std::cout << "stream_ms: min=" << stats.min_ms
              << " median=" << stats.median_ms
              << " mean=" << stats.mean_ms << "\n"
              << "aggregate_read_gib_per_s: min_time="
              << gib / (stats.min_ms / 1000.0)
              << " median_time=" << gib / (stats.median_ms / 1000.0)
              << "\n"
              << "aggregate_sampled_gib_per_s: min_time="
              << gib / kElementStride / (stats.min_ms / 1000.0)
              << " median_time="
              << gib / kElementStride / (stats.median_ms / 1000.0)
              << "\n";
    for (std::size_t node_index = 0; node_index < numa_nodes.size(); ++node_index) {
      const TimingStats node_stats = summarize(node_samples[node_index]);
      const double node_gib =
          bytes_per_node[node_index] / static_cast<double>(1ULL << 30);
      std::cout << "numa_node_" << numa_nodes[node_index]
                << "_read_gib_per_s: min_time="
                << node_gib / (node_stats.min_ms / 1000.0)
                << " median_time="
                << node_gib / (node_stats.median_ms / 1000.0) << "\n"
                << "numa_node_" << numa_nodes[node_index]
                << "_sampled_gib_per_s: min_time="
                << node_gib / kElementStride / (node_stats.min_ms / 1000.0)
                << " median_time="
                << node_gib / kElementStride / (node_stats.median_ms / 1000.0)
                << "\n";
    }
    std::cout << "checksum=0x" << std::hex << checksum << std::dec << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
