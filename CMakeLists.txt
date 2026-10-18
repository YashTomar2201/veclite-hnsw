cmake_minimum_required(VERSION 3.16)
project(veclite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release: Debug numbers are 10-50x slower and must never be reported.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_library(veclite STATIC src/hnsw.cpp src/io.cpp)
target_include_directories(veclite PUBLIC include)
target_compile_options(veclite PUBLIC -Wall -Wextra)

add_executable(flat_demo bench/flat_demo.cpp)
target_link_libraries(flat_demo PRIVATE veclite)

add_executable(bench_random bench/bench_random.cpp)
target_link_libraries(bench_random PRIVATE veclite)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE veclite)

enable_testing()
add_executable(tests tests/test_main.cpp tests/test_brute_force.cpp tests/test_flat_graph.cpp tests/test_hnsw.cpp tests/test_edge_cases.cpp)
target_include_directories(tests PRIVATE third_party/doctest)
target_link_libraries(tests PRIVATE veclite)
target_compile_definitions(tests PRIVATE VECLITE_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
add_test(NAME tests COMMAND tests)
