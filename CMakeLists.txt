cmake_minimum_required(VERSION 3.18)
project(gdr_copy LANGUAGES CXX)

if(WIN32)
    message(FATAL_ERROR "NIXL bench baseline is supported on Linux only.")
endif()

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
find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(NIXL QUIET IMPORTED_TARGET nixl)
endif()

if(NIXL_FOUND)
    set(NIXL_BENCH_TARGET PkgConfig::NIXL)
else()
    set(NIXL_ROOT "$ENV{NIXL_ROOT}" CACHE PATH "Path to the NIXL installation prefix")

    # UCX may be installed separately (e.g. built from source into ~/.local)
    set(UCX_ROOT "$ENV{UCX_ROOT}" CACHE PATH "Path to the UCX installation prefix")
    if(NOT UCX_ROOT)
        set(UCX_ROOT "${NIXL_ROOT}")
    endif()

    set(NIXL_LIB_HINTS
        ${NIXL_ROOT}/lib
        ${NIXL_ROOT}/lib64
        ${NIXL_ROOT}/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/local/lib64
        /usr/local/lib/x86_64-linux-gnu
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu)

    set(UCX_LIB_HINTS
        ${UCX_ROOT}/lib
        ${UCX_ROOT}/lib64
        ${UCX_ROOT}/lib/x86_64-linux-gnu
        ${NIXL_LIB_HINTS})

    find_path(NIXL_INCLUDE_DIR
        NAMES nixl.h
        HINTS ${NIXL_ROOT} /usr/local /usr
        PATH_SUFFIXES include)

    find_library(NIXL_CORE_LIB     NAMES nixl        HINTS ${NIXL_LIB_HINTS})
    find_library(NIXL_BUILD_LIB    NAMES nixl_build  HINTS ${NIXL_LIB_HINTS})
    find_library(NIXL_COMMON_LIB   NAMES nixl_common HINTS ${NIXL_LIB_HINTS})
    find_library(NIXL_STREAM_LIB   NAMES stream      HINTS ${NIXL_LIB_HINTS})
    find_library(NIXL_SERDES_LIB   NAMES serdes      HINTS ${NIXL_LIB_HINTS})
    # ucx_utils was removed in newer NIXL pip releases; treat as optional
    find_library(NIXL_UCXUTILS_LIB NAMES ucx_utils   HINTS ${NIXL_LIB_HINTS})
    find_library(UCX_UCP_LIB       NAMES ucp         HINTS ${UCX_LIB_HINTS})
    find_library(UCX_UCS_LIB       NAMES ucs         HINTS ${UCX_LIB_HINTS})
    find_library(UCX_UCT_LIB       NAMES uct         HINTS ${UCX_LIB_HINTS})

    if(NOT NIXL_INCLUDE_DIR OR
       NOT NIXL_CORE_LIB OR
       NOT NIXL_BUILD_LIB OR
       NOT NIXL_COMMON_LIB OR
       NOT NIXL_STREAM_LIB OR
       NOT NIXL_SERDES_LIB OR
       NOT UCX_UCP_LIB OR
       NOT UCX_UCS_LIB OR
       NOT UCX_UCT_LIB)
        message(FATAL_ERROR
            "NIXL not found. Install NIXL with pkg-config support, or set NIXL_ROOT "
            "to the NIXL install prefix that contains include/nixl.h and the NIXL/UCX libraries.")
    endif()

    add_library(nixl_external INTERFACE)
    target_include_directories(nixl_external INTERFACE
        ${NIXL_INCLUDE_DIR}
        ${NIXL_INCLUDE_DIR}/backend
        ${NIXL_INCLUDE_DIR}/utils/serdes
        ${NIXL_INCLUDE_DIR}/utils/common)
    target_link_libraries(nixl_external INTERFACE
        ${NIXL_CORE_LIB}
        ${NIXL_BUILD_LIB}
        ${NIXL_COMMON_LIB}
        ${NIXL_STREAM_LIB}
        ${NIXL_SERDES_LIB}
        $<$<BOOL:${NIXL_UCXUTILS_LIB}>:${NIXL_UCXUTILS_LIB}>
        ${UCX_UCP_LIB}
        ${UCX_UCS_LIB}
        ${UCX_UCT_LIB}
        dl
        rt
        pthread)
    set(NIXL_BENCH_TARGET nixl_external)
endif()

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
target_link_libraries(bench PRIVATE gdr_copy_static ${NIXL_BENCH_TARGET})
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
