#include <filesystem>
#include <format>
#include <iostream>

#include <graphy/EGRLoader.hpp>
#include <graphy/Graph.hpp>
#include <graphy/MST.hpp>

enum class Backends { baseline, SYCL };

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << std::format("Usage: {} <graph file> <backend>\n", argv[0]);
    return 1;
  }

  std::filesystem::path p = argv[1];
  if (!std::filesystem::exists(p)) {
    std::cerr << std::format("File {} does not exist\n", p.string());
    return 1;
  }

  Backends backend;
  if (std::string(argv[2]) == "baseline") {
    backend = Backends::baseline;
  } else if (std::string(argv[2]) == "SYCL") {
    backend = Backends::SYCL;
  } else {
    std::cerr << std::format("Unknown backend: {}\n", argv[2]);
    return 1;
  }

  p = std::filesystem::canonical(argv[1]);

  std::cerr << std::format("Reading graph from {}\n", p.string());

  auto g = graphy::loadFromEGR(p);

  if (backend == Backends::baseline) {
    std::cerr << "Using baseline backend\n";
    auto inMST = graphy::baseline::MST(g);
    for (int32_t i = 0; i < g.E; ++i) {
      std::cout << std::format("{}\n", inMST[i]);
    }
  } else {
    std::cerr << "Using SYCL backend\n";
    sycl::queue q;
    auto inMST = graphy::ECL_MST::syclMST(g, q);
    for (int32_t i = 0; i < g.E; ++i) {
      std::cout << std::format("{}\n", inMST[i]);
    }
  }

  return 0;
}
