// Run on Kunpeng with -mcpu=native to also execute the packed SVE BF16 path.
#include "indexer_packed_sve.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
  try {
    int cases = 0;
    for (int heads : {1, 3, 7, 16, 64}) {
      for (int dim : {1, 7, 17, 128}) {
        for (std::size_t tokens : {1, 3, 4, 7, 19}) {
          std::vector<std::uint16_t> q(heads * dim), keys(tokens * dim);
          std::vector<std::uint16_t> packed(((dim + 1) / 2) * heads * 2);
          std::vector<float> weights(heads), scores(tokens + 2);
          for (std::size_t i = 0; i < q.size(); ++i)
            q[i] = indexer::fp32_to_bf16(indexer::deterministic_float(i + 100000));
          for (std::size_t i = 0; i < keys.size(); ++i)
            keys[i] = indexer::fp32_to_bf16(indexer::deterministic_float(i + 1));
          for (int h = 0; h < heads; ++h)
            weights[h] = indexer::deterministic_float(h + 200000) * 4;
          indexer::pack_query(q.data(), packed.data(), heads, dim);
          for (int d = 0; d < dim + dim % 2; ++d)
            for (int h = 0; h < heads; ++h)
              if (packed[(d / 2 * heads + h) * 2 + d % 2] !=
                  (d < dim ? q[h * dim + d] : 0))
                throw std::runtime_error("query packing mismatch");
          for (bool use_packed : {false, true}) {
            std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
            scores.front() = scores.back() = 12345;
            indexer::score_range(q.data(), packed.data(), keys.data(), weights.data(),
                                  scores.data() + 1, tokens, heads, dim, use_packed);
            if (scores.front() != 12345 || scores.back() != 12345)
              throw std::runtime_error("output boundary overwritten");
            for (std::size_t token = 0; token < tokens; ++token) {
              double reference = 0;
              for (int h = 0; h < heads; ++h) {
                double dot = 0;
                for (int d = 0; d < dim; ++d)
                  dot += double(indexer::bf16_to_fp32(q[h * dim + d])) *
                         indexer::bf16_to_fp32(keys[token * dim + d]);
                reference += weights[h] * std::max(dot, 0.0);
              }
              if (!std::isfinite(scores[token + 1]) ||
                  std::abs(scores[token + 1] - reference) > 1.e-4 + 1.e-3 * std::abs(reference))
                throw std::runtime_error("score mismatch");
            }
            ++cases;
          }
        }
      }
    }
    std::cout << "pass cases=" << cases << " native=" << indexer::kernel_name()
#if INDEXER_HAS_SVE_BF16
              << " packed_sve=executed\n";
#else
              << " packed_sve=not_available (fallback and packing checked)\n";
#endif
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
