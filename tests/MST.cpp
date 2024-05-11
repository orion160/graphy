#include <format>
#include <iostream>
#include <memory>

#include <graphy/Graph.hpp>
#include <graphy/MST.hpp>

int main(int argc, char **argv) {
  graphy::CSRGraph g;
  g.V = 7;
  g.E = 22;
  g.N = std::make_unique<int32_t[]>(g.V + 1);
  g.N[0] = 0;
  g.N[1] = 2;
  g.N[2] = 3;
  g.N[3] = 7;
  g.N[4] = 12;
  g.N[5] = 16;
  g.N[6] = 20;
  g.N[7] = 22;
  g.F = std::make_unique<int32_t[]>(g.E);
  g.F[0] = 2;
  g.F[1] = 3;
  g.F[2] = 3;
  g.F[3] = 3;
  g.F[4] = 0;
  g.F[5] = 5;
  g.F[6] = 4;
  g.F[7] = 2;
  g.F[8] = 4;
  g.F[9] = 5;
  g.F[10] = 1;
  g.F[11] = 0;
  g.F[12] = 3;
  g.F[13] = 2;
  g.F[14] = 5;
  g.F[15] = 6;
  g.F[16] = 2;
  g.F[17] = 3;
  g.F[18] = 4;
  g.F[19] = 6;
  g.F[20] = 5;
  g.F[21] = 4;
  g.W = std::make_unique<int32_t[]>(g.E);
  g.W[0] = 3;
  g.W[1] = 7;
  g.W[2] = 1;
  g.W[3] = 1;
  g.W[4] = 3;
  g.W[5] = 3;
  g.W[6] = 2;
  g.W[7] = 1;
  g.W[8] = 9;
  g.W[9] = 4;
  g.W[10] = 1;
  g.W[11] = 7;
  g.W[12] = 9;
  g.W[13] = 2;
  g.W[14] = 2;
  g.W[15] = 8;
  g.W[16] = 3;
  g.W[17] = 4;
  g.W[18] = 2;
  g.W[19] = 11;
  g.W[20] = 11;
  g.W[21] = 8;

  auto inMST_CPU = graphy::baseline::MST(g);

  sycl::queue q;
  auto inMST_GPU = graphy::syclMST(g, q);

  bool match = true;
  for (int i = 0; i < g.E; ++i) {
    if (inMST_CPU[i] != inMST_GPU[i]) {
      match = false;
      break;
    }
  }

  if (match) {
    std::cout << "MST test passed\n";
    return 0;
  } else {
    std::cout << "MST test failed\n";
    return -1;
  }
}
