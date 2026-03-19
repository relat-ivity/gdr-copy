cmake_minimum_required(VERSION 3.18)
project(gdr_copy LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Aggressive optimisation for benchmark accuracy
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── RDMA (ibverbs) ────────────────────────────────────────────────────────────
find_library(IBVERBS_LIB NAMES ibverbs
             HINTS /usr/lib /usr/lib64 /usr/local/lib)
if(NOT IBVERBS_LIB)
    message(FATAL_ERROR
        "libibverbs not found.\n"
        "Install MLNX_OFED or run: apt install libibverbs-dev")
endif()
message(STATUS "Found ibverbs: ${IBVERBS_LIB}")

# ── NVML ─────────────────────────────────────────────────────────────────────
find_library(NVML_LIB NAMES nvidia-ml nvml
             HINTS ${CUDAToolkit_LIBRARY_DIR}
                   /usr/lib/x86_64-linux-gnu
                   /usr/lib64)
if(NOT NVML_LIB)
    message(WARNING "libnvml not found; GDR probe will skip NVML checks.")
endif()

# ── Include dirs ──────────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ── Library: gdr_copy (static) ────────────────────────────────────────────────
add_library(gdr_copy_static STATIC src/gdr_copy.cpp)
target_link_libraries(gdr_copy_static
    PUBLIC
        CUDA::cudart
        ${IBVERBS_LIB}
        pthread
)
if(NVML_LIB)
    target_link_libraries(gdr_copy_static PUBLIC ${NVML_LIB})
endif()

# ── Executable: demo ──────────────────────────────────────────────────────────
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE gdr_copy_static)
set_target_properties(demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# ── Executable: bench ─────────────────────────────────────────────────────────
add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE gdr_copy_static)
set_target_properties(bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS gdr_copy_static DESTINATION lib)
install(FILES include/gdr_copy.h DESTINATION include)

message(STATUS "")
message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA toolkit : ${CUDAToolkit_VERSION}")
message(STATUS "ibverbs      : ${IBVERBS_LIB}")
message(STATUS "")
message(STATUS "  cmake --build build --target demo   # basic correctness demo")
message(STATUS "  cmake --build build --target bench  # latency/BW sweep")
