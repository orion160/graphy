#include <algorithm>
#include <vector>

#include <graphy/MST.hpp>

namespace graphy {
namespace baseline {
static int32_t root(std::unique_ptr<int32_t[]> &parent, int32_t idx) {
  auto root = idx;
  auto next = parent[root];
  while (root != next) {
    parent[root] = parent[next];
    root = next;
    next = parent[next];
  }

  return root;
}

static void join(std::unique_ptr<int32_t[]> &parent, const int32_t u,
                 const int32_t v) {
  const int uRoot = root(parent, u);
  const int vRoot = root(parent, v);

  // Path compression
  if (uRoot > vRoot) {
    parent[vRoot] = uRoot;
  } else {
    parent[uRoot] = vRoot;
  }
}

struct EdgeInfo {
  int32_t w;
  int32_t idx;
  int32_t u;
  int32_t v;
};

std::unique_ptr<bool[]> MST(const CSRGraph &g) {
  auto inMST = std::make_unique<bool[]>(g.E);
  auto parent = std::make_unique<int32_t[]>(g.V);

  std::fill(inMST.get(), inMST.get() + g.E, false);
  for (size_t i = 0; i < g.V; ++i) {
    parent[i] = i;
  }

  std::vector<EdgeInfo> edges;
  for (int32_t i = 0; i < g.V; i++) {
    for (int32_t j = g.N[i]; j < g.N[i + 1]; ++j) {
      const int n = g.F[j];
      if (n > i) {
        edges.emplace_back(EdgeInfo{g.W[j], j, i, n});
      }
    }
  }

  std::sort(edges.begin(), edges.end(),
            [](const EdgeInfo &lhs, const EdgeInfo &rhs) -> bool {
              return lhs.w < rhs.w;
            });

  int count = g.V - 1;
  for (int p = 0; p < edges.size(); ++p) {
    const auto u = edges[p].u;
    const auto v = edges[p].v;

    const auto uRoot = root(parent, u);
    const auto vRoot = root(parent, v);
    if (uRoot != vRoot) {
      const auto j = edges[p].idx;
      inMST[j] = true;
      join(parent, uRoot, vRoot);

      count--;

      if (count == 0) {
        break;
      }
    }
  }

  return inMST;
}
} // namespace baseline
} // namespace graphy
