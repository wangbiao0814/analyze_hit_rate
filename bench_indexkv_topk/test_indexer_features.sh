#!/usr/bin/env bash
# Preprocessor-only tests: synthetic target macros must never be used to build
# executable code. No Arm hardware or cross-compilation toolchain is required.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compiler=${CXX:-c++}
cases=0
probe() {
  local expected=$1
  shift
  printf '#include "%s/indexer_features.h"\n#if INDEXER_HAS_SVE_BF16 != %s\n#error incorrect SVE BF16 feature detection\n#endif\n' \
    "$script_dir" "$expected" \
    | "$compiler" -undef -E -x c++ "$@" -o /dev/null -
  cases=$((cases + 1))
}
probe 0
probe 1 -D__ARM_FEATURE_SVE_BF16=1
probe 1 -D__GNUC__=12 -D__ARM_FEATURE_SVE=1 \
  -D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1 -D__ARM_FEATURE_SVE_BITS=0
probe 0 -D__GNUC__=12 -D__ARM_FEATURE_SVE=1
probe 0 -D__GNUC__=12 -D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1
probe 0 -D__GNUC__=4 -D__clang__=1 -D__ARM_FEATURE_SVE=1 \
  -D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1
probe 1 -D__GNUC__=4 -D__clang__=1 -D__ARM_FEATURE_SVE_BF16=1
probe 0 -D__ARM_FEATURE_SVE_BF16=0
probe 0 -D__GNUC__=12 -D__ARM_FEATURE_SVE=0 \
  -D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=1
probe 0 -D__GNUC__=12 -D__ARM_FEATURE_SVE=1 \
  -D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC=0
printf 'pass feature-detection cases=%d\n' "$cases"
