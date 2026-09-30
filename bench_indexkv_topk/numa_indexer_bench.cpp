// Persistent pthread NUMA indexer: one global sequence, node-local K/Q, and
// disjoint writes into one contiguous FP32 logits array. Linux only.
#include "indexer_packed_sve.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#include <vector>

#if !defined(__linux__)
#error "numa_indexer_bench requires Linux"
#endif

namespace {
using Clock = std::chrono::steady_clock;
using indexer::deterministic_float;
using indexer::fp32_to_bf16;
constexpr std::size_t kTokensPerLine = 64 / sizeof(float);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "lock-free counters required");
static_assert(std::atomic<bool>::is_always_lock_free, "lock-free flags required");

struct Options {
  std::size_t seq_len = 131072;
  int heads = 64, dim = 128, warmup = 10, iters = 100;
  int controller_cpu = -1;
  std::uint64_t target_us = 75;
  std::string cpus, kernel = "auto", huge_pages = "thp";
  bool bind_memory = true, refresh_query = true, check = false, check_all = false;
};

std::uint64_t number(const std::string& text, bool zero = false) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("invalid integer: " + text);
  std::size_t used = 0;
  const auto value = std::stoull(text, &used);
  if (used != text.size() || (!zero && value == 0))
    throw std::runtime_error("invalid integer: " + text);
  return value;
}

int small_number(const std::string& text, bool zero = false) {
  const auto value = number(text, zero);
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("integer too large: " + text);
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i == argc) throw std::runtime_error("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--cpus") o.cpus = value();
    else if (arg == "--seq-len") o.seq_len = number(value());
    else if (arg == "--heads") o.heads = small_number(value());
    else if (arg == "--dim") o.dim = small_number(value());
    else if (arg == "--warmup") o.warmup = small_number(value(), true);
    else if (arg == "--iters") o.iters = small_number(value());
    else if (arg == "--target-us") o.target_us = number(value());
    else if (arg == "--controller-cpu") o.controller_cpu = small_number(value(), true);
    else if (arg == "--kernel") o.kernel = value();
    else if (arg == "--huge-pages") o.huge_pages = value();
    else if (arg == "--memory-policy") {
      const auto v = value();
      if (v != "bind" && v != "first-touch") throw std::runtime_error("memory-policy: bind or first-touch");
      o.bind_memory = v == "bind";
    } else if (arg == "--query-mode") {
      const auto v = value();
      if (v != "refresh" && v != "static") throw std::runtime_error("query-mode: refresh or static");
      o.refresh_query = v == "refresh";
    } else if (arg == "--check") o.check = true;
    else if (arg == "--check-all") o.check = o.check_all = true;
    else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " --cpus LIST [options]\n"
          "  --seq-len 131072 --heads 64 --dim 128\n"
          "  --kernel auto|native|packed-sve\n"
          "  --memory-policy bind|first-touch (default: bind, requires mbind)\n"
          "  --huge-pages off|thp|2m|1g (K only; default: thp)\n"
          "  --query-mode refresh|static (default: refresh)\n"
          "  --warmup 10 --iters 100 --target-us 75\n"
          "  --controller-cpu N (optional; choose a spare physical core)\n"
          "  --check / --check-all (sampled / every token vs FP64 reference)\n";
      std::exit(0);
    } else throw std::runtime_error("unknown option: " + arg);
  }
  if (o.cpus.empty()) throw std::runtime_error("--cpus is required; one logical CPU per physical core recommended");
  if (o.kernel != "auto" && o.kernel != "native" && o.kernel != "packed-sve")
    throw std::runtime_error("kernel: auto, native, or packed-sve");
  if (o.huge_pages != "off" && o.huge_pages != "thp" && o.huge_pages != "2m" && o.huge_pages != "1g")
    throw std::runtime_error("huge-pages: off, thp, 2m, or 1g");
  return o;
}

std::vector<int> parse_cpus(const std::string& text) {
  std::vector<int> cpus;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed))
    throw std::runtime_error("sched_getaffinity failed");
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto comma = text.find(',', pos);
    const auto part = text.substr(pos, comma == std::string::npos ? comma : comma - pos);
    const auto dash = part.find('-');
    const int first = small_number(part.substr(0, dash), true);
    const int last = dash == std::string::npos ? first : small_number(part.substr(dash + 1), true);
    if (first > last || last >= CPU_SETSIZE) throw std::runtime_error("invalid CPU range: " + part);
    for (int cpu = first; cpu <= last; ++cpu) {
      if (!CPU_ISSET(cpu, &allowed)) throw std::runtime_error("CPU outside allowed affinity: " + std::to_string(cpu));
      if (std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()) throw std::runtime_error("duplicate CPU");
      cpus.push_back(cpu);
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
    if (pos == text.size()) throw std::runtime_error("trailing comma in --cpus");
  }
  return cpus;
}

int cpu_node(int cpu) {
  if (access("/sys/devices/system/node", F_OK)) return 0;
  for (int node = 0; node < CPU_SETSIZE; ++node) {
    const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(node);
    if (access(path.c_str(), F_OK) == 0) return node;
  }
  throw std::runtime_error("cannot discover NUMA node for CPU " + std::to_string(cpu));
}

void pin(int cpu) {
  if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("invalid CPU for pinning");
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (error) throw std::runtime_error("pin CPU " + std::to_string(cpu) + ": " + std::strerror(error));
}

std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::runtime_error("size overflow");
  return a * b;
}

// Overflow-safe balanced partition, including a possible final partial tile.
std::size_t split(std::size_t total, std::size_t rank, std::size_t count) {
  return total / count * rank + std::min(rank, total % count);
}

struct Mapping {
  void* data = nullptr;
  std::size_t bytes = 0;
  Mapping() = default;
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  ~Mapping() { if (data) munmap(data, bytes); }
  void allocate(std::size_t requested, int node, bool bind, const std::string& pages) {
    if (!requested) return;
    const long base_page = sysconf(_SC_PAGESIZE);
    if (base_page <= 0) throw std::runtime_error("cannot query page size");
    std::size_t page = static_cast<std::size_t>(base_page);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (pages == "2m" || pages == "1g") {
      const unsigned shift = pages == "2m" ? 21 : 30;
      page = std::size_t{1} << shift;
      flags |= MAP_HUGETLB | static_cast<int>(shift << 26);
    }
    if (requested > std::numeric_limits<std::size_t>::max() - page + 1) throw std::runtime_error("mapping size overflow");
    bytes = (requested + page - 1) / page * page;
    void* result = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (result == MAP_FAILED) throw std::runtime_error(std::string("mmap: ") + std::strerror(errno));
    data = result;
    if (bind) {
      const std::size_t bits = sizeof(unsigned long) * 8;
      std::vector<unsigned long> mask(static_cast<std::size_t>(node) / bits + 1, 0);
      mask[node / bits] |= 1UL << (node % bits);
      if (syscall(SYS_mbind, data, bytes, MPOL_BIND, mask.data(),
                  static_cast<unsigned long>(node + 1), 0UL) != 0)
        throw std::runtime_error("mbind node " + std::to_string(node) + ": " + std::strerror(errno) +
                                 "; use --memory-policy first-touch only if strict binding is unavailable");
    }
    const int advice = pages == "thp" ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
    if ((pages == "thp" || pages == "off") && madvise(data, bytes, advice))
      throw std::runtime_error(std::string("madvise: ") + std::strerror(errno));
  }
  template<class T> T* as() { return static_cast<T*>(data); }
};

void relax(std::size_t& polls) {
#if defined(__aarch64__)
  asm volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause");
#endif
  if (++polls % 4096 == 0) sched_yield();
}

struct Node {
  int id = -1;
  std::size_t begin = 0, end = 0;
  std::vector<std::size_t> workers;
  Mapping keys, query, packed_query, weights;
  std::string error;
  alignas(64) std::atomic<bool> ready{false};
  alignas(64) std::atomic<std::uint64_t> query_generation{0};
};

struct Context {
  Options options;
  bool packed = false;
  float* scores = nullptr;
  std::vector<std::uint16_t> queries[2];
  std::vector<float> weights[2];
  alignas(64) std::atomic<std::uint64_t> generation{0};
  alignas(64) std::atomic<bool> stop{false};
};

struct Worker {
  alignas(64) std::atomic<std::uint64_t> done{0};
  alignas(64) std::atomic<bool> ready{false};
  alignas(64) Context* context = nullptr;
  Node* node = nullptr;
  int cpu = -1;
  bool leader = false;
  std::size_t begin = 0, end = 0;
  std::string error;
  Clock::time_point begin_time, end_time;
};

void update_query(Context& c, Node& node, int bank) {
  const auto& q = c.queries[bank];
  std::memcpy(node.query.data, q.data(), q.size() * sizeof(q[0]));
  std::memcpy(node.weights.data, c.weights[bank].data(), c.weights[bank].size() * sizeof(float));
  if (c.packed) indexer::pack_query(node.query.as<std::uint16_t>(), node.packed_query.as<std::uint16_t>(),
                                  c.options.heads, c.options.dim);
}

void* worker_main(void* arg) {
  Worker& w = *static_cast<Worker*>(arg);
  Context& c = *w.context;
  Node& n = *w.node;
  const Options& o = c.options;
  try {
    pin(w.cpu);
    if (w.leader) {
      const auto q_bytes = product(product(o.heads, o.dim), sizeof(std::uint16_t));
      n.keys.allocate(product(product(n.end - n.begin, o.dim), sizeof(std::uint16_t)), n.id, o.bind_memory, o.huge_pages);
      n.query.allocate(q_bytes, n.id, o.bind_memory, "off");
      n.weights.allocate(product(o.heads, sizeof(float)), n.id, o.bind_memory, "off");
      if (c.packed) n.packed_query.allocate(product(product((std::size_t(o.dim) + 1) / 2, o.heads), 4), n.id, o.bind_memory, "off");
      update_query(c, n, 0);
    }
  } catch (const std::exception& e) {
    w.error = e.what();
    if (w.leader) n.error = w.error;
  }
  if (w.leader) n.ready.store(true, std::memory_order_release);
  std::size_t polls = 0;
  while (!n.ready.load(std::memory_order_acquire)) {
    if (c.stop.load(std::memory_order_relaxed)) return nullptr;
    relax(polls);
  }
  if (w.error.empty() && n.error.empty()) {
    auto* keys = n.keys.as<std::uint16_t>();
    for (std::size_t token = w.begin; token < w.end; ++token) {
      for (int d = 0; d < o.dim; ++d) {
        keys[(token - n.begin) * o.dim + d] = fp32_to_bf16(
            deterministic_float(token * o.dim + d + 1));
      }
      c.scores[token] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  std::uint64_t observed = 0;
  w.ready.store(true, std::memory_order_release);
  for (;;) {
    std::uint64_t generation;
    polls = 0;
    while ((generation = c.generation.load(std::memory_order_acquire)) == observed) {
      if (c.stop.load(std::memory_order_relaxed)) return nullptr;
      relax(polls);
    }
    if (c.stop.load(std::memory_order_relaxed)) return nullptr;
    observed = generation;
    if (c.options.refresh_query) {
      if (w.leader) {
        update_query(c, n, static_cast<int>(generation & 1));
        n.query_generation.store(generation, std::memory_order_release);
      } else {
        polls = 0;
        while (n.query_generation.load(std::memory_order_acquire) != generation) {
          if (c.stop.load(std::memory_order_relaxed)) return nullptr;
          relax(polls);
        }
      }
    }
    w.begin_time = Clock::now();
    indexer::score_range(n.query.as<std::uint16_t>(), n.packed_query.as<std::uint16_t>(),
                         n.keys.as<std::uint16_t>() + (w.begin - n.begin) * o.dim,
                         n.weights.as<float>(), c.scores + w.begin,
                         w.end - w.begin, o.heads, o.dim, c.packed);
    w.end_time = Clock::now();
    w.done.store(generation, std::memory_order_release);
  }
}

struct Pool {
  Context& context;
  std::vector<pthread_t> handles;
  std::size_t created = 0;
  Pool(Context& c, std::size_t count) : context(c), handles(count) {}
  void stop() {
    context.stop.store(true, std::memory_order_relaxed);
    for (std::size_t i = 0; i < created; ++i) pthread_join(handles[i], nullptr);
    created = 0;
  }
  ~Pool() { stop(); }
};

double us(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}
struct Stats { double minimum, median, p95, maximum; };
Stats stats(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto n = samples.size();
  return {samples.front(), n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2,
          samples[static_cast<std::size_t>(std::ceil(n * .95)) - 1], samples.back()};
}
void print_stats(const char* name, const Stats& s) {
  std::cout << name << "_us: min=" << s.minimum << " median=" << s.median
            << " p95=" << s.p95 << " max=" << s.maximum << '\n';
}

void check(Context& c, const std::vector<std::unique_ptr<Worker>>& workers,
           std::uint64_t generation) {
  const Options& o = c.options;
  const int bank = o.refresh_query ? static_cast<int>(generation & 1) : 0;
  std::vector<std::size_t> tokens;
  if (o.check_all) {
    tokens.resize(o.seq_len);
    std::iota(tokens.begin(), tokens.end(), std::size_t{0});
  } else {
    for (const auto& w : workers) {
      tokens.push_back(w->begin);
      tokens.push_back((w->begin + w->end) / 2);
      tokens.push_back(w->end - 1);
    }
    for (std::size_t i = 0; i < std::min<std::size_t>(o.seq_len, 256); ++i)
      tokens.push_back(i * (o.seq_len - 1) / std::max<std::size_t>(1, std::min<std::size_t>(o.seq_len, 256) - 1));
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
  }
  double max_error = 0;
  for (auto token : tokens) {
    double reference = 0;
    for (int head = 0; head < o.heads; ++head) {
      double dot = 0;
      for (int d = 0; d < o.dim; ++d) {
        const float k = indexer::bf16_to_fp32(fp32_to_bf16(deterministic_float(token * o.dim + d + 1)));
        dot += double(indexer::bf16_to_fp32(c.queries[bank][std::size_t(head) * o.dim + d])) * k;
      }
      reference += c.weights[bank][head] * std::max(dot, 0.0);
    }
    const double error = std::abs(double(c.scores[token]) - reference);
    max_error = std::max(max_error, error);
    if (!std::isfinite(c.scores[token]) || error > 1.e-4 + 1.e-3 * std::abs(reference))
      throw std::runtime_error("score mismatch at global token " + std::to_string(token));
  }
  std::cout << "check=pass checked_tokens=" << tokens.size() << " max_abs_error=" << max_error << '\n';
}

int run(int argc, char** argv) {
  Context c;
  c.options = parse_options(argc, argv);
  const Options& o = c.options;
  const auto cpus = parse_cpus(o.cpus);
#if defined(__ARM_FEATURE_SVE_BF16)
  c.packed = o.kernel != "native";
#else
  if (o.kernel == "packed-sve") throw std::runtime_error("packed-sve requires a build with SVE BF16 support");
#endif
  product(product(o.seq_len, o.dim), sizeof(std::uint16_t));
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::unique_ptr<Worker>> workers;
  for (int cpu : cpus) {
    const int id = cpu_node(cpu);
    auto found = std::find_if(nodes.begin(), nodes.end(), [id](const auto& n) { return n->id == id; });
    if (found == nodes.end()) {
      nodes.emplace_back(new Node);
      nodes.back()->id = id;
      found = nodes.end() - 1;
    }
    auto worker = std::make_unique<Worker>();
    worker->context = &c;
    worker->node = found->get();
    worker->cpu = cpu;
    worker->leader = (*found)->workers.empty();
    (*found)->workers.push_back(workers.size());
    workers.push_back(std::move(worker));
  }
  // Partition whole output cache lines: global 128K tokens are NOT replicated
  // on each node. For tiny tests the final tile can contain fewer than 16 tokens.
  const std::size_t tiles = (o.seq_len - 1) / kTokensPerLine + 1;
  for (std::size_t rank = 0; rank < nodes.size(); ++rank) {
    Node& n = *nodes[rank];
    const auto first = split(tiles, rank, nodes.size());
    const auto last = split(tiles, rank + 1, nodes.size());
    if (last - first < n.workers.size()) throw std::runtime_error("too many workers for sequence: need at least one 16-token tile per worker on every node");
    n.begin = std::min(o.seq_len, first * kTokensPerLine);
    n.end = std::min(o.seq_len, last * kTokensPerLine);
    for (std::size_t j = 0; j < n.workers.size(); ++j) {
      Worker& w = *workers[n.workers[j]];
      w.begin = std::min(o.seq_len, (first + split(last - first, j, n.workers.size())) * kTokensPerLine);
      w.end = std::min(o.seq_len, (first + split(last - first, j + 1, n.workers.size())) * kTokensPerLine);
    }
  }
  for (int bank = 0; bank < 2; ++bank) {
    c.queries[bank].resize(product(o.heads, o.dim));
    c.weights[bank].resize(o.heads);
    for (std::size_t i = 0; i < c.queries[bank].size(); ++i)
      c.queries[bank][i] = fp32_to_bf16(deterministic_float(i + 0x100000000ULL + bank * 0x400000000ULL));
    for (int h = 0; h < o.heads; ++h)
      c.weights[bank][h] = deterministic_float(h + 0x200000000ULL + bank * 0x400000000ULL) * 4;
  }
  Mapping output;
  output.allocate(product(o.seq_len, sizeof(float)), 0, false, "off");
  c.scores = output.as<float>(); // First-touched by pinned writers, never by main.
  if (o.controller_cpu >= 0) {
    (void)parse_cpus(std::to_string(o.controller_cpu)); // Respect the original allowed affinity.
    if (std::find(cpus.begin(), cpus.end(), o.controller_cpu) != cpus.end()) throw std::runtime_error("controller CPU must not also be a worker");
    pin(o.controller_cpu);
  }
  Pool pool(c, workers.size());
  for (; pool.created < workers.size(); ++pool.created) {
    const int error = pthread_create(&pool.handles[pool.created], nullptr, worker_main, workers[pool.created].get());
    if (error) throw std::runtime_error(std::string("pthread_create: ") + std::strerror(error));
  }
  for (const auto& w : workers) {
    std::size_t polls = 0;
    while (!w->ready.load(std::memory_order_acquire)) relax(polls);
  }
  for (const auto& w : workers) if (!w->error.empty()) throw std::runtime_error(w->error);
  for (const auto& n : nodes) if (!n->error.empty()) throw std::runtime_error(n->error);

  std::cout << std::fixed << std::setprecision(3)
            << "kernel=" << (c.packed ? "packed SVE BF16 4-token" : indexer::kernel_name()) << '\n'
            << "seq_len=" << o.seq_len << " heads=" << o.heads << " dim=" << o.dim << '\n'
            << "threads=" << workers.size() << " nodes=" << nodes.size() << " cpus=" << o.cpus << '\n'
            << "memory_policy=" << (o.bind_memory ? "bind" : "first-touch") << " huge_pages=" << o.huge_pages << '\n'
            << "query_mode=" << (o.refresh_query ? "refresh" : "static") << " cache_mode=repeated_resident_K\n"
            << "output=contiguous_fp32_first_touch target_basis=end_to_end_p95\n"
            << "controller_cpu=" << o.controller_cpu << " (-1=unbound)\n";
  for (const auto& n : nodes)
    std::cout << "node=" << n->id << " token_range=[" << n->begin << ',' << n->end << ") workers=" << n->workers.size()
              << " k_mib=" << double((n->end - n->begin) * o.dim * 2) / 1048576
              << " k_mapping_mib=" << double(n->keys.bytes) / 1048576 << '\n';

  std::uint64_t generation = 0;
  auto once = [&]() {
    const auto begin = Clock::now();
    c.generation.store(++generation, std::memory_order_release);
    for (const auto& w : workers) {
      std::size_t polls = 0;
      while (w->done.load(std::memory_order_acquire) != generation) relax(polls);
    }
    return us(begin, Clock::now());
  };
  for (int i = 0; i < o.warmup; ++i) once();
  std::vector<double> dispatch, kernel, skew;
  dispatch.reserve(o.iters); kernel.reserve(o.iters); skew.reserve(o.iters);
  for (int i = 0; i < o.iters; ++i) {
    dispatch.push_back(once());
    auto begin = workers.front()->begin_time, latest_begin = begin;
    auto end = workers.front()->end_time;
    for (const auto& w : workers) {
      begin = std::min(begin, w->begin_time);
      latest_begin = std::max(latest_begin, w->begin_time);
      end = std::max(end, w->end_time);
    }
    kernel.push_back(us(begin, end));
    skew.push_back(us(begin, latest_begin));
  }
  pool.stop();
  if (o.check) check(c, workers, generation);
  double checksum = 0;
  for (std::size_t token = 0; token < o.seq_len; ++token) {
    if (!std::isfinite(c.scores[token])) throw std::runtime_error("non-finite or unwritten logit");
    checksum += c.scores[token];
  }
  const auto ks = stats(kernel), ds = stats(dispatch);
  print_stats("kernel", ks); print_stats("end_to_end", ds); print_stats("worker_start_skew", stats(skew));
  const double flops = 2.0 * o.seq_len * o.heads * o.dim;
  std::cout << "effective_tflops=" << flops / (ks.median * 1.e6)
            << " unique_k_gib_per_s=" << double(o.seq_len * o.dim * 2) / (1ULL << 30) / (ks.median * 1.e-6) << '\n'
            << "target_us=" << o.target_us << " required_tflops=" << flops / (o.target_us * 1.e6) << '\n'
            << "target_end_to_end_p95=" << (ds.p95 <= o.target_us ? "pass" : "miss")
            << " within_target_iterations=" << std::count_if(dispatch.begin(), dispatch.end(), [&](double t) { return t <= o.target_us; })
            << '/' << o.iters << " correctness=" << (o.check ? "checked" : "not_checked") << '\n'
            << "checksum=" << std::setprecision(9) << checksum << '\n';
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& e) { std::cerr << "fatal: " << e.what() << '\n'; return 1; }
}
