#include <cstdint>
#include <format>
#include <iostream>
#include <memory>

#include <graphy/MST/ECL_MST.hpp>

bool DS_root_test_1(sycl::queue q) {
  auto parent = std::make_unique<int32_t[]>(16);
  parent[0] = 9;
  parent[1] = 0;
  parent[2] = 2;
  parent[3] = 3;
  parent[4] = 3;
  parent[5] = 4;
  parent[6] = 5;
  parent[7] = 0;
  parent[8] = 6;
  parent[9] = 9;
  parent[10] = 6;
  parent[11] = 10;
  parent[12] = 1;
  parent[13] = 5;
  parent[14] = 3;
  parent[15] = 12;

  auto result = std::make_unique<int32_t[]>(16);
  {
    sycl::buffer parentD(parent.get(), {16});
    sycl::buffer<int32_t> resultD(16);
    resultD.set_final_data(result.get());
    q.submit([&](sycl::handler &h) {
      auto r = resultD.get_access(h, sycl::write_only, sycl::no_init);
      auto p = parentD.get_access(h, sycl::read_only);
      h.parallel_for(16, [=](sycl::id<1> i) {
        auto idx = i[0];
        r[i] = graphy::ECL_MST::DS::root(idx, p);
      });
    });
  }

  auto expected = std::make_unique<int32_t[]>(16);
  expected[0] = 9;
  expected[1] = 9;
  expected[2] = 2;
  expected[3] = 3;
  expected[4] = 3;
  expected[5] = 3;
  expected[6] = 3;
  expected[7] = 9;
  expected[8] = 3;
  expected[9] = 9;
  expected[10] = 3;
  expected[11] = 3;
  expected[12] = 9;
  expected[13] = 3;
  expected[14] = 3;
  expected[15] = 9;

  for (uint32_t i = 0; i < 16; ++i) {
    if (result[i] != expected[i]) {

      std::cerr << std::format("Mismatch -> result[{}]:{} != expected[{}]:{}\n",
                               i, result[i], i, expected[i]);
      return false;
    }
  }

  return true;
}

int main() {
  sycl::queue q{sycl::cpu_selector_v};
  if (!DS_root_test_1(q)) {
    std::cerr << "DS_root_test_1 failed\n";
    return 1;
  } else {
    std::cout << "DS_root_test_1 passed\n";
  }
}
