#pragma once

#include <filesystem>

#include <graphy/Graph.hpp>

namespace graphy {
[[deprecated]] CSRGraph loadFromEGR(std::filesystem::path p);
}
