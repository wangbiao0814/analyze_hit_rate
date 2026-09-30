// Standalone CPU benchmark for the DeepSeek-V3.2 DSA indexer score.
//
// The score computed for every historical token s is:
//   score[s] = sum_h weight[h] * relu(dot(q[h, :], index_k[s, :]))
//
// q and index_k are stored as BF16. Dot products accumulate in FP32.  On Arm,
// the benchmark uses SVE BF16 BFDOT when available, then NEON BF16 BFDOT, and
// finally a portable scalar BF16-to-FP32 implementation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "indexer_bf16_kernels.h"

#if INDEXER_HAS_SVE_BF16
#include <arm_sve.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::int64_t seq_len = 128 * 1024;
  int heads = 64;
  int dim = 128;
  int threads = 1;
  int warmup = 3;
  int iters = 10;
  bool check = false;
  bool memory_only = false;
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
      << "  --seq-len N   historical tokens (default: 131072)\n"
      << "  --heads N     index heads (default: 64)\n"
      << "  --dim N       index head dimension (default: 128)\n"
      << "  --threads N   pthread workers over seq_len (default: 1)\n"
      << "  --warmup N    warmup iterations (default: 3)\n"
      << "  --iters N     measured iterations (default: 10)\n"
      << "  --memory-only stream index-K once with a lightweight XOR reduction\n"
      << "  --check       compare the first 256 scores with scalar code\n"
      << "  --help        show this message\n";
  std::exit(error.empty() ? 0 : 2);
}

template <typename T>
T parse_integer(const char* text, const char* name, bool allow_zero = false) {
  std::size_t consumed = 0;
  const std::string value(text);
  long long parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception&) {
    usage("cpu_indexer_topk_bench", std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || parsed < (allow_zero ? 0 : 1) ||
      parsed > static_cast<long long>(std::numeric_limits<T>::max())) {
    usage("cpu_indexer_topk_bench", std::string("invalid ") + name + ": " + value);
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

    if (arg == "--seq-len") {
      options.seq_len = parse_integer<std::int64_t>(require_value("--seq-len"), "seq-len");
    } else if (arg == "--heads") {
      options.heads = parse_integer<int>(require_value("--heads"), "heads");
    } else if (arg == "--dim") {
      options.dim = parse_integer<int>(require_value("--dim"), "dim");
    } else if (arg == "--threads") {
      options.threads =
          parse_integer<int>(require_value("--threads"), "threads");
    } else if (arg == "--warmup") {
      options.warmup =
          parse_integer<int>(require_value("--warmup"), "warmup", true);
    } else if (arg == "--iters") {
      options.iters = parse_integer<int>(require_value("--iters"), "iters");
    } else if (arg == "--memory-only") {
      options.memory_only = true;
    } else if (arg == "--check") {
      options.check = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0], "unknown option: " + arg);
    }
  }
  return options;
}

using namespace indexer;

const char* memory_kernel_name() {
#if defined(__aarch64__)
  return "Arm NEON memory-stream XOR";
#else
  return "portable memory-stream XOR";
#endif
}

std::uint16_t xor_key_scalar(const std::uint16_t* key, int dim) {
  std::uint16_t result = 0;
  for (int d = 0; d < dim; ++d) {
    result ^= key[d];
  }
  return result;
}

#if defined(__aarch64__)
std::uint16_t xor_key_native(const std::uint16_t* key, int dim) {
  uint16x8_t accum0 = vdupq_n_u16(0);
  uint16x8_t accum1 = vdupq_n_u16(0);
  uint16x8_t accum2 = vdupq_n_u16(0);
  uint16x8_t accum3 = vdupq_n_u16(0);
  int d = 0;
  for (; d + 32 <= dim; d += 32) {
    accum0 = veorq_u16(accum0, vld1q_u16(key + d));
    accum1 = veorq_u16(accum1, vld1q_u16(key + d + 8));
    accum2 = veorq_u16(accum2, vld1q_u16(key + d + 16));
    accum3 = veorq_u16(accum3, vld1q_u16(key + d + 24));
  }
  uint16x8_t accum = veorq_u16(
      veorq_u16(accum0, accum1), veorq_u16(accum2, accum3));
  for (; d + 8 <= dim; d += 8) {
    accum = veorq_u16(accum, vld1q_u16(key + d));
  }

  const uint64x2_t lanes = vreinterpretq_u64_u16(accum);
  std::uint64_t folded = vgetq_lane_u64(lanes, 0) ^ vgetq_lane_u64(lanes, 1);
  folded ^= folded >> 32u;
  folded ^= folded >> 16u;
  std::uint16_t result = static_cast<std::uint16_t>(folded);
  for (; d < dim; ++d) {
    result ^= key[d];
  }
  return result;
}
#else
std::uint16_t xor_key_native(const std::uint16_t* key, int dim) {
  return xor_key_scalar(key, dim);
}
#endif

void initialize_inputs(std::vector<std::uint16_t>& q,
                       std::vector<std::uint16_t>& index_k,
                       std::vector<float>& weights) {
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(index_k.size()); ++i) {
    index_k[static_cast<std::size_t>(i)] =
        fp32_to_bf16(deterministic_float(static_cast<std::uint64_t>(i) + 1));
  }

  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = fp32_to_bf16(deterministic_float(i + 0x100000000ULL));
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    // Indexer head gates are not constrained to be positive.
    weights[i] = deterministic_float(i + 0x200000000ULL) * 4.0f;
  }
}

struct WorkerArgs {
  const std::uint16_t* q;
  const std::uint16_t* index_k;
  const float* weights;
  float* scores;
  std::size_t token_begin;
  std::size_t token_end;
  int heads;
  int dim;
  bool memory_only;
};

void* compute_scores_worker(void* opaque) {
  const WorkerArgs& args = *static_cast<WorkerArgs*>(opaque);
  for (std::size_t token = args.token_begin; token < args.token_end; ++token) {
    const std::uint16_t* key = args.index_k + token * args.dim;
    if (args.memory_only) {
      args.scores[token] = static_cast<float>(xor_key_native(key, args.dim));
    } else {
      args.scores[token] = weighted_relu_sum_native(
          args.q, key, args.weights, args.heads, args.dim);
    }
  }
  return nullptr;
}

void compute_scores_parallel(const std::vector<std::uint16_t>& q,
                             const std::vector<std::uint16_t>& index_k,
                             const std::vector<float>& weights,
                             std::vector<float>& scores,
                             int heads,
                             int dim,
                             int requested_threads,
                             bool memory_only) {
  const std::size_t worker_count = std::min<std::size_t>(
      static_cast<std::size_t>(requested_threads), scores.size());
  const std::size_t tokens_per_worker = scores.size() / worker_count;

  std::vector<pthread_t> workers(worker_count);
  std::vector<WorkerArgs> args(worker_count);
  std::size_t created = 0;
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    const std::size_t token_begin = worker * tokens_per_worker;
    // The final worker also handles seq_len % worker_count tokens.
    const std::size_t token_end = worker + 1 == worker_count
                                      ? scores.size()
                                      : token_begin + tokens_per_worker;
    args[worker] = {q.data(),
                    index_k.data(),
                    weights.data(),
                    scores.data(),
                    token_begin,
                    token_end,
                    heads,
                    dim,
                    memory_only};

    const int error = pthread_create(
        &workers[worker], nullptr, compute_scores_worker, &args[worker]);
    if (error != 0) {
      for (std::size_t join_worker = 0; join_worker < created; ++join_worker) {
        pthread_join(workers[join_worker], nullptr);
      }
      throw std::runtime_error(
          std::string("pthread_create failed: ") + std::strerror(error));
    }
    ++created;
  }

  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    const int error = pthread_join(workers[worker], nullptr);
    if (error != 0) {
      throw std::runtime_error(
          std::string("pthread_join failed: ") + std::strerror(error));
    }
  }
}

TimingStats summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::runtime_error("cannot summarize an empty timing sample");
  }
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
                            ? (samples[middle - 1] + samples[middle]) * 0.5
                            : samples[middle];
  return {samples.front(), median, mean};
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_scores(const std::vector<std::uint16_t>& q,
                  const std::vector<std::uint16_t>& index_k,
                  const std::vector<float>& weights,
                  const std::vector<float>& scores,
                  int heads,
                  int dim) {
  const std::size_t check_tokens = std::min<std::size_t>(256, scores.size());
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (std::size_t token = 0; token < check_tokens; ++token) {
    const std::uint16_t* key = index_k.data() + token * dim;
    float reference = 0.0f;
    for (int head = 0; head < heads; ++head) {
      const float dot = dot_bf16_scalar(q.data() + static_cast<std::int64_t>(head) * dim,
                                        key,
                                        dim);
      reference += weights[head] * std::max(dot, 0.0f);
    }
    const double abs_error = std::abs(static_cast<double>(scores[token]) - reference);
    const double rel_error = abs_error / std::max(1.0e-6, std::abs(static_cast<double>(reference)));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }
  std::cout << "check_tokens=" << check_tokens << " max_abs_error=" << max_abs_error
            << " max_rel_error=" << max_rel_error << "\n";
  if (max_abs_error > 5.0e-3 && max_rel_error > 5.0e-3) {
    throw std::runtime_error("SVE/NEON score check failed");
  }
}

void check_memory_scores(const std::vector<std::uint16_t>& index_k,
                         const std::vector<float>& scores,
                         int dim) {
  const std::size_t check_tokens = std::min<std::size_t>(256, scores.size());
  for (std::size_t token = 0; token < check_tokens; ++token) {
    const std::uint16_t* key = index_k.data() + token * dim;
    const float expected = static_cast<float>(xor_key_scalar(key, dim));
    if (scores[token] != expected) {
      throw std::runtime_error("memory-only score check failed");
    }
  }
  std::cout << "check_tokens=" << check_tokens << " memory_check=pass\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    const std::size_t q_elements =
        static_cast<std::size_t>(options.heads) * options.dim;
    const std::size_t k_elements =
        static_cast<std::size_t>(options.seq_len) * options.dim;
    std::vector<std::uint16_t> q(q_elements);
    std::vector<std::uint16_t> index_k(k_elements);
    std::vector<float> weights(options.heads);
    std::vector<float> scores(static_cast<std::size_t>(options.seq_len));

    initialize_inputs(q, index_k, weights);

    std::cout << "kernel="
              << (options.memory_only ? memory_kernel_name() : kernel_name())
              << "\n";
#if INDEXER_HAS_SVE_BF16
    std::cout << "sve_bits=" << svcntb() * 8 << "\n";
#endif
    if (options.memory_only) {
      std::cout << "shape: index_k=[" << options.seq_len << ',' << options.dim
                << "] heads_ignored=" << options.heads << "\n";
    } else {
      std::cout << "shape: q=[" << options.heads << ',' << options.dim
                << "] index_k=[" << options.seq_len << ',' << options.dim << "]\n";
    }
    std::cout << "index_k_size_mib=" << std::fixed << std::setprecision(2)
              << (index_k.size() * sizeof(std::uint16_t) / 1048576.0) << "\n";
    std::cout << "threads=" << options.threads << "\n";

    auto run_kernel = [&]() {
      compute_scores_parallel(q,
                              index_k,
                              weights,
                              scores,
                              options.heads,
                              options.dim,
                              options.threads,
                              options.memory_only);
    };

    for (int i = 0; i < options.warmup; ++i) {
      run_kernel();
    }

    std::vector<double> score_samples;
    score_samples.reserve(options.iters);

    for (int i = 0; i < options.iters; ++i) {
      const auto score_begin = Clock::now();
      run_kernel();
      const auto score_end = Clock::now();
      score_samples.push_back(elapsed_ms(score_begin, score_end));
    }

    if (options.check) {
      if (options.memory_only) {
        check_memory_scores(index_k, scores, options.dim);
      } else {
        check_scores(q, index_k, weights, scores, options.heads, options.dim);
      }
    }

    const TimingStats score_stats = summarize(score_samples);
    const double flops = 2.0 * options.seq_len * options.heads * options.dim;
    const double effective_gflops = flops / (score_stats.median_ms * 1.0e6);
    const double unique_k_gib = index_k.size() * sizeof(std::uint16_t) /
                                static_cast<double>(1ULL << 30);
    const double unique_k_gib_per_s = unique_k_gib / (score_stats.median_ms / 1000.0);

    auto print_stats = [](const char* name, const TimingStats& stats) {
      std::cout << name << "_ms: min=" << stats.min_ms << " median=" << stats.median_ms
                << " mean=" << stats.mean_ms << "\n";
    };
    print_stats(options.memory_only ? "memory_stream" : "score", score_stats);
    if (options.memory_only) {
      std::cout << "index_k_read_gib_per_s=" << unique_k_gib_per_s << "\n";
    } else {
      std::cout << "score_effective_gflops=" << effective_gflops << "\n";
      std::cout << "unique_index_k_gib_per_s=" << unique_k_gib_per_s << "\n";
    }

    const double checksum =
        std::accumulate(scores.begin(), scores.end(), 0.0);
    std::cout << "checksum=" << checksum << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << "\n";
    return 1;
  }
}
