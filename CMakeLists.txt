cmake_minimum_required(VERSION 3.28)
project(graphy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("${SYCL_IMPLEMENTATION}" STREQUAL "Intel OneAPI")
    find_package(IntelSYCL REQUIRED)
endif()

add_library(graphy)
target_sources(graphy
    PUBLIC
    FILE_SET mst_hdrs TYPE HEADERS
    BASE_DIRS .
)

add_sycl_to_target(TARGET graphy)

add_subdirectory(graphy)
add_subdirectory(tests)
add_subdirectory(utils)
