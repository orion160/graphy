#include <graphy/Graph.hpp>

#include <sycl/sycl.hpp>

namespace graphy::ECL_MST {
namespace DS {
SYCL_EXTERNAL int32_t root(const int32_t node,
             sycl::accessor<int32_t, 1, sycl::access::mode::read> parent);

SYCL_EXTERNAL void join(int32_t u, int32_t v,
          sycl::accessor<int32_t, 1, sycl::access::mode::read_write> parent);
} // namespace DS

std::unique_ptr<bool[]> syclMST(const CSRGraph &g, sycl::queue &q);
} // namespace graphy::ECL_MST
