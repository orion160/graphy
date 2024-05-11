#pragma once

#include <cstdint>
#include <memory>

namespace graphy {
struct CSRGraph {
  int32_t V;
  int32_t E;
  std::unique_ptr<int32_t[]> N;
  std::unique_ptr<int32_t[]> F;
  std::unique_ptr<int32_t[]> W;
};
} // namespace graphy
