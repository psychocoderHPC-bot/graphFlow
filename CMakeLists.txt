#
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
cmake_minimum_required(VERSION 3.25)

project(
    graphFlow
    VERSION 0.1.0
    DESCRIPTION "Host-scheduled asynchronous chunk-scheduling graph on alpaka3"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GRAPHFLOW_ENABLE_CUDA "Compile the CUDA backend (sets alpaka_DEP_CUDA=ON)" OFF)
option(GRAPHFLOW_ENABLE_HIP "Compile the HIP backend (sets alpaka_DEP_HIP=ON)" OFF)
option(GRAPHFLOW_ENABLE_OPENMP "Enable the OpenMP executor" OFF)
option(GRAPHFLOW_BUILD_TESTS "Build the Catch2 test suite" ON)
option(GRAPHFLOW_BUILD_EXAMPLES "Build the particle-feedback example" ON)

set(
    GRAPHFLOW_ALPAKA_SOURCE_DIR
    "/workspace"
    CACHE PATH "Path to the pinned alpaka3 source tree"
)

if(NOT EXISTS "${GRAPHFLOW_ALPAKA_SOURCE_DIR}/include/alpaka/alpaka.hpp")
    message(
        FATAL_ERROR
        "GRAPHFLOW_ALPAKA_SOURCE_DIR='${GRAPHFLOW_ALPAKA_SOURCE_DIR}' does not contain include/alpaka/alpaka.hpp"
    )
endif()

if(GRAPHFLOW_ENABLE_CUDA AND GRAPHFLOW_ENABLE_HIP)
    message(
        FATAL_ERROR
        "GRAPHFLOW_ENABLE_CUDA and GRAPHFLOW_ENABLE_HIP are mutually exclusive in one build; use one build "
        "directory per backend"
    )
endif()

# Accelerator languages and the matching alpaka dependency options must be established before
# add_subdirectory(alpaka) so alpaka builds its CUDA/HIP targets and alpaka_finalize() compiles
# the graphFlow sources for that backend. The CPU backend stays enabled in accelerator builds.
include(CheckLanguage)

if(GRAPHFLOW_ENABLE_CUDA)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "GRAPHFLOW_ENABLE_CUDA=ON but no CUDA compiler was found. Install the CUDA toolkit (nvcc) or "
            "set CMAKE_CUDA_COMPILER."
        )
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures for graphFlow" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(alpaka_DEP_CUDA ON CACHE BOOL "Enable the CUDA dependency in alpaka" FORCE)
    set(alpaka_CUDA_NvidiaGpu ON CACHE BOOL "Enable api::cuda/deviceKind::nvidiaGpu in alpaka" FORCE)
endif()

if(GRAPHFLOW_ENABLE_HIP)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(
            FATAL_ERROR
            "GRAPHFLOW_ENABLE_HIP=ON but no HIP compiler was found. Install ROCm (amdclang++) or set "
            "CMAKE_HIP_COMPILER."
        )
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx90a CACHE STRING "HIP architectures for graphFlow" FORCE)
    endif()
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
    set(alpaka_DEP_HIP ON CACHE BOOL "Enable the HIP dependency in alpaka" FORCE)
    set(alpaka_HIP_AmdGpu ON CACHE BOOL "Enable api::hip/deviceKind::amdGpu in alpaka" FORCE)
endif()

# alpaka3 is consumed from the pinned source tree; the project only uses its public API.
add_subdirectory("${GRAPHFLOW_ALPAKA_SOURCE_DIR}" alpaka)

add_library(graphFlow INTERFACE)
add_library(graphFlow::graphFlow ALIAS graphFlow)
target_compile_features(graphFlow INTERFACE cxx_std_20)
target_include_directories(
    graphFlow
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(graphFlow INTERFACE alpaka::alpaka)

if(GRAPHFLOW_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(graphFlow INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

if(GRAPHFLOW_BUILD_EXAMPLES)
    add_executable(graphFlow_particle_feedback examples/particle_feedback/main.cpp)
    target_link_libraries(graphFlow_particle_feedback PRIVATE graphFlow::graphFlow)
    target_include_directories(
        graphFlow_particle_feedback
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/particle_feedback
    )
    alpaka_finalize(graphFlow_particle_feedback)
endif()

if(GRAPHFLOW_BUILD_TESTS)
    include(CTest)
    enable_testing()

    find_package(Catch2 3 REQUIRED)

    add_executable(
        graphFlow_tests
        tests/unit/test_reservation.cpp
        tests/unit/test_backend.cpp
        tests/unit/test_cli_contract.cpp
        tests/unit/test_chunk_pool.cpp
        tests/unit/test_dense_bin_table.cpp
        tests/unit/test_philox.cpp
        tests/integration/test_dense_bins.cpp
        tests/integration/test_hash_bins.cpp
        tests/integration/test_particle_feedback.cpp
        tests/integration/test_scripted_routing.cpp
        tests/integration/test_pressure.cpp
        tests/integration/test_failure_paths.cpp
    )
    if(GRAPHFLOW_ENABLE_CUDA)
        target_sources(graphFlow_tests PRIVATE tests/unit/test_cuda_kernel.cpp)
    endif()
    target_link_libraries(graphFlow_tests PRIVATE graphFlow::graphFlow Catch2::Catch2WithMain)
    target_include_directories(
        graphFlow_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/particle_feedback
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference
    )

    alpaka_finalize(graphFlow_tests)

    include(Catch)
    catch_discover_tests(graphFlow_tests)
endif()
