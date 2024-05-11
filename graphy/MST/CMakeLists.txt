target_sources(graphy
    PRIVATE
    MST.cpp ECL_MST.cpp
    PUBLIC
    FILE_SET mst_hdrs
    FILES
    ECL_MST.hpp
)
