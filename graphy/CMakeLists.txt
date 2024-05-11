target_sources(graphy
    PRIVATE
    EGRLoader.cpp
    PUBLIC
    FILE_SET mst_hdrs
    FILES
    Graph.hpp MST.hpp EGRLoader.hpp
)

add_subdirectory(MST)
