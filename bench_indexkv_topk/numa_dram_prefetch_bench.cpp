// Linux/AArch64 streaming-read benchmark for estimating the DRAM bandwidth
// available to a CPU-side indexer. Each pthread is pinned to one CPU and
// first-touches its own buffer, so choosing CPUs from multiple NUMA nodes
// allocates and reads memory locally on those nodes under the default Linux
// first-touch policy.

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

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if !defined(__linux__)
#error "numa_dram_prefetch_bench requires Linux"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  int threads = 1;
  std::size_t buffer_mib = 256;
  std::size_t prefetch_distance = 0;
  int warmup = 1;
  int iters = 5;
  std::string cpu_list;
  bool request_thp = true;
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
      << "  --buffer-mib N           MiB read by each worker (default: 256)\n"
      << "  --prefetch-distance N    bytes ahead for AArch64 PLDL3KEEP; 0 disables\n"
      << "  --warmup N               untimed scans (default: 1)\n"
      << "  --iters N                measured scans (default: 5)\n"
      << "  --no-thp                 do not request transparent huge pages\n"
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
    } else if (arg == "--prefetch-distance") {
      options.prefetch_distance = parse_integer<std::size_t>(
          require_value("--prefetch-distance"), "prefetch-distance", true);
    } else if (arg == "--warmup") {
      options.warmup =
          parse_integer<int>(require_value("--warmup"), "warmup", true);
    } else if (arg == "--iters") {
      options.iters = parse_integer<int>(require_value("--iters"), "iters");
    } else if (arg == "--no-thp") {
      options.request_thp = false;
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
  return options;
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
  std::size_t offset = 0;
#if defined(__aarch64__)
  uint64x2_t accum0 = vdupq_n_u64(0);
  uint64x2_t accum1 = vdupq_n_u64(0);
  uint64x2_t accum2 = vdupq_n_u64(0);
  uint64x2_t accum3 = vdupq_n_u64(0);
  for (; offset + 64 <= bytes; offset += 64) {
    if (prefetch_distance != 0 && prefetch_distance < bytes - offset) {
      prefetch_l3(data + offset + prefetch_distance);
    }
    const std::uint64_t* words =
        reinterpret_cast<const std::uint64_t*>(data + offset);
    accum0 = veorq_u64(accum0, vld1q_u64(words));
    accum1 = veorq_u64(accum1, vld1q_u64(words + 2));
    accum2 = veorq_u64(accum2, vld1q_u64(words + 4));
    accum3 = veorq_u64(accum3, vld1q_u64(words + 6));
  }
  const uint64x2_t accum = veorq_u64(
      veorq_u64(accum0, accum1), veorq_u64(accum2, accum3));
  std::uint64_t checksum = vgetq_lane_u64(accum, 0) ^ vgetq_lane_u64(accum, 1);
#else
  std::uint64_t checksum = 0;
  for (; offset + 64 <= bytes; offset += 64) {
    if (prefetch_distance != 0 && prefetch_distance < bytes - offset) {
      prefetch_l3(data + offset + prefetch_distance);
    }
    for (std::size_t lane = 0; lane < 64; lane += sizeof(std::uint64_t)) {
      std::uint64_t value;
      std::memcpy(&value, data + offset + lane, sizeof(value));
      checksum ^= value;
    }
  }
#endif
  for (; offset + sizeof(std::uint64_t) <= bytes;
       offset += sizeof(std::uint64_t)) {
    std::uint64_t value;
    std::memcpy(&value, data + offset, sizeof(value));
    checksum ^= value;
  }
  for (; offset < bytes; ++offset) {
    checksum ^= data[offset];
  }
  return checksum;
}

struct SharedState {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  int ready = 0;
  int finished = 0;
  std::uint64_t generation = 0;
  bool stop = false;
};

struct alignas(64) WorkerState {
  SharedState* shared = nullptr;
  std::uint8_t* buffer = nullptr;
  std::size_t bytes = 0;
  std::size_t prefetch_distance = 0;
  int cpu = -1;
  int affinity_error = 0;
  std::uint64_t checksum = 0;
};

void* worker_main(void* opaque) {
  WorkerState& worker = *static_cast<WorkerState*>(opaque);
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(worker.cpu, &affinity);
  worker.affinity_error = pthread_setaffinity_np(
      pthread_self(), sizeof(affinity), &affinity);

  // This write happens after pinning. Under the default Linux memory policy it
  // physically allocates every page on the NUMA node local to worker.cpu.
  std::memset(worker.buffer, worker.cpu + 1, worker.bytes);

  SharedState& shared = *worker.shared;
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

    worker.checksum +=
        stream_read(worker.buffer, worker.bytes, worker.prefetch_distance) +
        observed_generation;

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

    const std::size_t bytes_per_worker = options.buffer_mib * 1024ULL * 1024ULL;
    if (bytes_per_worker >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(options.threads)) {
      throw std::runtime_error("aggregate working set is too large");
    }
    const std::size_t aggregate_bytes =
        bytes_per_worker * static_cast<std::size_t>(options.threads);

    SharedState shared;
    if (pthread_mutex_init(&shared.mutex, nullptr) != 0 ||
        pthread_cond_init(&shared.condition, nullptr) != 0) {
      throw std::runtime_error("failed to initialize pthread synchronization");
    }

    std::vector<WorkerState> states(static_cast<std::size_t>(options.threads));
    std::vector<pthread_t> workers(static_cast<std::size_t>(options.threads));
    for (int worker = 0; worker < options.threads; ++worker) {
      void* mapping = mmap(nullptr,
                           bytes_per_worker,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1,
                           0);
      if (mapping == MAP_FAILED) {
        throw std::runtime_error(
            std::string("mmap failed: ") + std::strerror(errno));
      }
      if (options.request_thp) {
        (void)madvise(mapping, bytes_per_worker, MADV_HUGEPAGE);
      }
      states[worker].shared = &shared;
      states[worker].buffer = static_cast<std::uint8_t*>(mapping);
      states[worker].bytes = bytes_per_worker;
      states[worker].prefetch_distance = options.prefetch_distance;
      states[worker].cpu = cpus[worker];
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
      for (const WorkerState& worker : states) {
        if (worker.buffer != nullptr) {
          munmap(worker.buffer, worker.bytes);
        }
      }
    };

    pthread_mutex_lock(&shared.mutex);
    while (shared.ready != options.threads) {
      pthread_cond_wait(&shared.condition, &shared.mutex);
    }
    pthread_mutex_unlock(&shared.mutex);

    for (const WorkerState& worker : states) {
      if (worker.affinity_error != 0) {
        const std::string message =
            "failed to pin a worker to CPU " + std::to_string(worker.cpu) +
            ": " + std::strerror(worker.affinity_error);
        stop_workers();
        release_buffers();
        pthread_cond_destroy(&shared.condition);
        pthread_mutex_destroy(&shared.mutex);
        throw std::runtime_error(
            message);
      }
    }

    std::cout << "workers=" << options.threads << "\n"
              << "cpus=" << join_cpus(cpus) << "\n"
              << "buffer_mib_per_worker=" << options.buffer_mib << "\n"
              << "aggregate_working_set_gib=" << std::fixed
              << std::setprecision(3)
              << aggregate_bytes / static_cast<double>(1ULL << 30) << "\n"
              << "prefetch="
              << (options.prefetch_distance == 0 ? "disabled" : "AArch64 PLDL3KEEP")
              << "\n"
              << "prefetch_distance_bytes=" << options.prefetch_distance << "\n";

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
    for (int i = 0; i < options.iters; ++i) {
      samples.push_back(run_once());
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
              << "checksum=0x" << std::hex << checksum << std::dec << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
