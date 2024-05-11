#pragma once

#include <memory>

#include <graphy/Graph.hpp>
#include <graphy/MST/ECL_MST.hpp>

namespace graphy {
namespace baseline {
std::unique_ptr<bool[]> MST(const CSRGraph &g);
} // namespace baseline
} // namespace graphy
