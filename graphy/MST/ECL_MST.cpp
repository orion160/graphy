#include <format>
#include <iostream>

#include <graphy/MST/ECL_MST.hpp>

namespace graphy {
int32_t root(const int32_t node,
             sycl::accessor<int32_t, 1, sycl::access::mode::read> parent) {
  auto root = node;
  while (root != parent[root]) {
    root = parent[root];
  }

  return root;
}

void join(int32_t u, int32_t v,
          sycl::accessor<int32_t, 1, sycl::access::mode::read_write> parent) {
  int32_t M = u;
  int32_t m = v;

  do {
    M = sycl::max(M, m);
    m = sycl::min(M, m);
  } while (
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>{parent[M]}
          .compare_exchange_weak(M, m));
}
/** ECL MST **/
struct WorkList {
  int32_t sourceVertex;
  int32_t destinationVertex;
  int32_t weight;
  int32_t edgeID;
};

class InitializeVertices {
public:
  InitializeVertices(const int32_t nodes,
                     sycl::accessor<int32_t, 1, sycl::access::mode::write> p,
                     sycl::accessor<int64_t, 1, sycl::access::mode::write> mv)
      : nodes{nodes}, parent{p}, minv{mv} {}

  void operator()(sycl::item<1> it) const {
    const auto node = it[0];
    if (node < nodes) {
      parent[node] = node;
      minv[node] = std::numeric_limits<int64_t>::max();
    }
  }

private:
  const int32_t nodes;
  sycl::accessor<int32_t, 1, sycl::access::mode::write> parent;
  sycl::accessor<int64_t, 1, sycl::access::mode::write> minv;
};

class InitializeWorkList {
public:
  InitializeWorkList(
      const int32_t nodes,
      sycl::accessor<int32_t, 1, sycl::access::mode::read_write> wlSZ,
      sycl::accessor<WorkList, 1, sycl::access::mode::write> wl,
      sycl::accessor<int32_t, 1, sycl::access::mode::read> n,
      sycl::accessor<int32_t, 1, sycl::access::mode::read> f,
      sycl::accessor<int32_t, 1, sycl::access::mode::read> w)
      : nodes{nodes}, wlSZ{wlSZ}, wl{wl}, N{n}, F{f}, W{w} {}

  void operator()(sycl::item<1> it) const {
    const auto node = it[0];

    if (node < nodes) {
      const auto adjBegin = N[node];
      const auto adjEnd = N[node + 1];
      const auto degree = adjEnd - adjBegin;

      for (int32_t eID = adjBegin; eID < adjEnd; ++eID) {
        const auto neighbor = F[eID];
        if (node < neighbor) {
          const auto w = W[eID];
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              k{wlSZ[0]};
          const auto idx = k.fetch_add(1);

          wl[idx] = WorkList{static_cast<int32_t>(node), neighbor, w, eID};
        }
      }
    }
  }

private:
  const int32_t nodes;
  sycl::accessor<int32_t, 1, sycl::access::mode::read_write> wlSZ;
  sycl::accessor<WorkList, 1, sycl::access::mode::write> wl;
  sycl::accessor<int32_t, 1, sycl::access::mode::read> N;
  sycl::accessor<int32_t, 1, sycl::access::mode::read> F;
  sycl::accessor<int32_t, 1, sycl::access::mode::read> W;
};

class GatherLightestNode {
public:
  GatherLightestNode(
      sycl::accessor<int32_t, 1, sycl::access::mode::read> wl1SZ,
      sycl::accessor<WorkList, 1, sycl::access::mode::read> wl1,
      sycl::accessor<int32_t, 1, sycl::access::mode::read_write> wl2SZ,
      sycl::accessor<WorkList, 1, sycl::access::mode::write> wl2,
      sycl::accessor<int32_t, 1, sycl::access::mode::read> parent,
      sycl::accessor<int64_t, 1, sycl::access::mode::read_write> minv)
      : wl1SZ{wl1SZ}, wl1{wl1}, wl2SZ{wl2SZ}, wl2{wl2}, parent{parent},
        minv{minv} {}

  void operator()(sycl::id<1> id) const {
    const auto idx = id[0];

    if (idx < wl1SZ[0]) {
      auto wItem = wl1[idx];
      const auto u = root(wItem.sourceVertex, parent);
      const auto v = root(wItem.destinationVertex, parent);

      if (u != v) {
        wItem.sourceVertex = u;
        wItem.destinationVertex = v;

        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            k{wl2SZ[0]};
        wl2[k.fetch_add(1)] = wItem;
        const auto value = static_cast<int64_t>(wItem.weight) << 32 |
                           static_cast<int64_t>(wItem.edgeID);

        if (value < minv[u]) {
          sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              m{minv[u]};
          m.fetch_min(value);
        }
        if (value < minv[v]) {
          sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              m{minv[v]};
          m.fetch_min(value);
        }
      }
    }
  }

private:
  sycl::accessor<int32_t, 1, sycl::access::mode::read> wl1SZ;
  sycl::accessor<WorkList, 1, sycl::access::mode::read> wl1;
  sycl::accessor<int32_t, 1, sycl::access::mode::read_write> wl2SZ;
  sycl::accessor<WorkList, 1, sycl::access::mode::write> wl2;
  sycl::accessor<int32_t, 1, sycl::access::mode::read> parent;
  sycl::accessor<int64_t, 1, sycl::access::mode::read_write> minv;
};

class InMST {
public:
  InMST(sycl::accessor<int32_t, 1, sycl::access::mode::read> wlSZ,
        sycl::accessor<WorkList, 1, sycl::access::mode::read> wl,
        sycl::accessor<int64_t, 1, sycl::access::mode::read> minv,
        sycl::accessor<bool, 1, sycl::access::mode::write> inMST,
        sycl::accessor<int32_t, 1, sycl::access::mode::read_write> parent)
      : wlSZ{wlSZ}, wl{wl}, minv{minv}, inMST{inMST}, parent{parent} {}

  void operator()(sycl::id<1> id) const {
    const auto idx = id[0];

    if (idx < wlSZ[0]) {
      const auto wItem = wl[idx];
      const auto value = static_cast<int64_t>(wItem.weight) << 32 |
                         static_cast<int64_t>(wItem.edgeID);
      if (value == minv[wItem.sourceVertex] ||
          value == minv[wItem.destinationVertex]) {
        join(wItem.sourceVertex, wItem.destinationVertex, parent);
        inMST[wItem.edgeID] = true;
      }
    }
  }

private:
  sycl::accessor<int32_t, 1, sycl::access::mode::read> wlSZ;
  sycl::accessor<WorkList, 1, sycl::access::mode::read> wl;
  sycl::accessor<int32_t, 1, sycl::access::mode::read_write> parent;
  sycl::accessor<int64_t, 1, sycl::access::mode::read> minv;
  sycl::accessor<bool, 1, sycl::access::mode::write> inMST;
};

class ResetMinV {
public:
  ResetMinV(sycl::accessor<int32_t, 1, sycl::access::mode::read> wlSZ,
            sycl::accessor<WorkList, 1, sycl::access::mode::read> worklist,
            sycl::accessor<int64_t, 1, sycl::access::mode::write> minv)
      : wlSZ{wlSZ}, worklist{worklist}, minv{minv} {}

  void operator()(sycl::id<1> id) const {
    const auto idx = id[0];

    if (idx < wlSZ[0]) {
      const auto wItem = worklist[idx];
      minv[wItem.sourceVertex] = std::numeric_limits<int64_t>::max();
      minv[wItem.destinationVertex] = std::numeric_limits<int64_t>::max();
    }
  }

private:
  sycl::accessor<int32_t, 1, sycl::access::mode::read> wlSZ;
  sycl::accessor<WorkList, 1, sycl::access::mode::read> worklist;
  sycl::accessor<int64_t, 1, sycl::access::mode::write> minv;
};

std::unique_ptr<bool[]> syclMST(const CSRGraph &g, sycl::queue &q) {
  auto r = std::make_unique<bool[]>(g.E);

  {
    sycl::buffer<int32_t, 1> parent(sycl::range<1>(g.V));
    sycl::buffer<int64_t, 1> minV(sycl::range<1>(g.V));
    q.submit([&](sycl::handler &h) {
      auto p = parent.get_access(h, sycl::write_only, sycl::no_init);
      auto mv = minV.get_access(h, sycl::write_only, sycl::no_init);
      h.parallel_for(sycl::range<1>(g.V), InitializeVertices{g.V, p, mv});
    });

    sycl::buffer<int32_t, 1> gN(sycl::range<1>(g.V + 1));
    q.submit([&](sycl::handler &h) {
      auto n = gN.get_access(h, sycl::write_only, sycl::no_init);
      h.copy(g.N.get(), n);
    });

    sycl::buffer<int32_t, 1> gF(sycl::range<1>(g.E));
    q.submit([&](sycl::handler &h) {
      auto f = gF.get_access(h, sycl::write_only, sycl::no_init);
      h.copy(g.F.get(), f);
    });

    sycl::buffer<int32_t, 1> gW(sycl::range<1>(g.E));
    q.submit([&](sycl::handler &h) {
      auto w = gW.get_access(h, sycl::write_only, sycl::no_init);
      h.copy(g.W.get(), w);
    });

    sycl::buffer<int32_t, 1> worklist1SZ(sycl::range<1>(1));
    q.submit([&](sycl::handler &h) {
      auto sz = worklist1SZ.get_access(h, sycl::write_only, sycl::no_init);
      h.single_task([=]() { sz[0] = 0; });
    });

    sycl::buffer<WorkList, 1> worklist1(sycl::range<1>(g.E / 2));
    q.submit([&](sycl::handler &h) {
      auto wlSZ = worklist1SZ.get_access(h, sycl::read_write);
      auto wl = worklist1.get_access(h, sycl::write_only, sycl::no_init);

      auto N = gN.get_access(h, sycl::read_only);
      auto F = gF.get_access(h, sycl::read_only);
      auto W = gW.get_access(h, sycl::read_only);

      h.parallel_for(sycl::range<1>(g.V),
                     InitializeWorkList{g.V, wlSZ, wl, N, F, W});
    });

    sycl::buffer<bool, 1> inMST(sycl::range<1>(g.E));
    inMST.set_final_data(r.get());
    q.submit([&](sycl::handler &h) {
      auto in = inMST.get_access(h, sycl::write_only, sycl::no_init);
      h.fill(in, false);
    });

    int32_t wlSZ;
    {
      auto sz = worklist1SZ.get_host_access(sycl::read_only);
      wlSZ = sz[0];
    }

    std::cerr << "wlSZ: " << wlSZ << std::endl;

    sycl::buffer<WorkList, 1> worklist2(sycl::range<1>(g.E / 2));
    sycl::buffer<int32_t, 1> worklist2SZ(sycl::range<1>(1));
    while (wlSZ > 0) {
      q.submit([&](sycl::handler &h) {
        auto sz = worklist2SZ.get_access(h, sycl::write_only, sycl::no_init);
        h.single_task([=]() { sz[0] = 0; });
      });

      int32_t wlSubmit;
      {
        auto sz = worklist1SZ.get_host_access(sycl::read_only);
        wlSubmit = sz[0];
      }

      q.submit([&](sycl::handler &h) {
        auto wl1SZ = worklist1SZ.get_access(h, sycl::read_only);
        auto wl1 = worklist1.get_access(h, sycl::read_only);
        auto wl2SZ = worklist2SZ.get_access(h, sycl::read_write);
        auto wl2 = worklist2.get_access(h, sycl::write_only, sycl::no_init);
        auto p = parent.get_access(h, sycl::read_only);
        auto mv = minV.get_access(h, sycl::read_write);

        h.parallel_for(sycl::range<1>(wlSubmit),
                       GatherLightestNode{wl1SZ, wl1, wl2SZ, wl2, p, mv});
      });

      std::swap(worklist1, worklist2);
      std::swap(worklist1SZ, worklist2SZ);

      {
        auto sz = worklist1SZ.get_host_access(sycl::read_only);
        wlSZ = sz[0];
      }

      std::cerr << "wlSZ: " << wlSZ << std::endl;

      if (wlSZ > 0) {
        q.submit([&](sycl::handler &h) {
          auto wl1SZ = worklist1SZ.get_access(h, sycl::read_only);
          auto wl1 = worklist1.get_access(h, sycl::read_only);
          auto mv = minV.get_access(h, sycl::read_only);
          auto in = inMST.get_access(h, sycl::write_only);
          auto p = parent.get_access(h, sycl::read_write);

          h.parallel_for(sycl::range<1>(wlSubmit),
                         InMST{wl1SZ, wl1, mv, in, p});
        });

        q.submit([&](sycl::handler &h) {
          auto wl1SZ = worklist1SZ.get_access(h, sycl::read_only);
          auto wl1 = worklist1.get_access(h, sycl::read_only);
          auto mv = minV.get_access(h, sycl::write_only);

          h.parallel_for(sycl::range<1>(wlSubmit), ResetMinV{wl1SZ, wl1, mv});
        });
      }
    }
  }

  return r;
}
} // namespace graphy
