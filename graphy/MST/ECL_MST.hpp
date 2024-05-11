#include <graphy/Graph.hpp>

#include <sycl/sycl.hpp>

namespace graphy {
std::unique_ptr<bool[]> syclMST(const CSRGraph &g, sycl::queue &q);
}
