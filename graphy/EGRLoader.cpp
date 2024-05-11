#include <fstream>

#include <graphy/EGRLoader.hpp>

namespace graphy {
CSRGraph loadFromEGR(std::filesystem ::path p) {
  std::ifstream file(p, std::ios::binary);

  CSRGraph g;
  file.read(reinterpret_cast<char *>(&g.V), sizeof(g.V));
  file.read(reinterpret_cast<char *>(&g.E), sizeof(g.E));

  g.N = std::make_unique<int32_t[]>(g.V + 1);
  file.read(reinterpret_cast<char *>(g.N.get()), sizeof(int32_t) * (g.V + 1));

  g.F = std::make_unique<int32_t[]>(g.E);
  file.read(reinterpret_cast<char *>(g.F.get()), sizeof(int32_t) * g.E);

  g.W = std::make_unique<int32_t[]>(g.E);
  file.read(reinterpret_cast<char *>(g.W.get()), sizeof(int32_t) * g.E);

  if (file.eof() && file.fail()) {
    g.W.reset();
  } else if (file.gcount() != sizeof(int32_t) * g.E) {
    throw std::runtime_error("Failed to read edge weights");
  }

  return g;
}
} // namespace graphy
