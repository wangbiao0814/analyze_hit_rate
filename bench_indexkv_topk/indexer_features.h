#pragma once

// Older GCC releases provide SVE BF16 intrinsics without defining the combined
// ACLE macro. Use GCC's separate target features in that case. Keep this header
// independent of system headers so feature detection can be regression-tested.
// https://gcc.gnu.org/gcc-15/changes.html
// Do not synthesize reserved __ARM_* compiler macros or fix the SVE vector size.
#if (defined(__ARM_FEATURE_SVE_BF16) && __ARM_FEATURE_SVE_BF16) || \
    (defined(__GNUC__) && !defined(__clang__) && \
     defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE && \
     defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && \
     __ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#define INDEXER_HAS_SVE_BF16 1
#else
#define INDEXER_HAS_SVE_BF16 0
#endif
