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

    add_executable(graphFlow_photon_scattering examples/photon_scattering/main.cpp)
    target_link_libraries(graphFlow_photon_scattering PRIVATE graphFlow::graphFlow)
    target_include_directories(
        graphFlow_photon_scattering
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/photon_scattering
    )
    alpaka_finalize(graphFlow_photon_scattering)
endif()

if(GRAPHFLOW_BUILD_TESTS)
    include(CTest)
    enable_testing()

    find_package(Catch2 3 REQUIRED)

    add_executable(
        graphFlow_tests
        tests/unit/test_reservation.cpp
        tests/unit/test_backend.cpp
        tests/unit/test_launch_policy.cpp
        tests/unit/test_cli_contract.cpp
        tests/unit/test_chunk_pool.cpp
        tests/unit/test_payload_storage.cpp
        tests/unit/test_payload_slot_storage.cpp
        tests/unit/test_flow_builder.cpp
        tests/unit/test_flow_execution.cpp
        tests/unit/test_flow_failure.cpp
        tests/unit/test_pool_budget.cpp
        tests/unit/test_dense_bin_table.cpp
        tests/unit/test_route_table.cpp
        tests/unit/test_graph_builder.cpp
        tests/unit/test_philox.cpp
        tests/integration/test_dense_bins.cpp
        tests/integration/test_hash_bins.cpp
        tests/integration/test_particle_feedback.cpp
        tests/integration/test_scripted_routing.cpp
        tests/integration/test_pressure.cpp
        tests/integration/test_failure_paths.cpp
        tests/integration/test_batching.cpp
        tests/integration/test_framespec.cpp
        tests/integration/test_generic_dag.cpp
        tests/integration/test_generic_flow.cpp
        tests/integration/test_photon_scattering.cpp
    )
    if(GRAPHFLOW_ENABLE_CUDA)
        target_sources(
            graphFlow_tests
            PRIVATE
                tests/unit/test_cuda_kernel.cpp
                tests/integration/test_flow_execution_cuda.cpp
                tests/integration/test_photon_scattering_cuda.cpp
        )
    endif()
    if(GRAPHFLOW_ENABLE_HIP)
        # Compile-only: no AMD device is available, so this adds no CTest case.
        target_sources(graphFlow_tests PRIVATE tests/integration/test_flow_execution_hip.cpp)
    endif()
    target_link_libraries(graphFlow_tests PRIVATE graphFlow::graphFlow Catch2::Catch2WithMain)
    target_include_directories(
        graphFlow_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/particle_feedback
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/photon_scattering
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference
    )

    alpaka_finalize(graphFlow_tests)

    include(Catch)
    catch_discover_tests(graphFlow_tests)

    # Source-level contract: the user-facing example must not fall back to numeric node ids,
    # route vectors, manual runner/pool construction, or the deprecated example stages.
    add_test(
        NAME example.public_api
        COMMAND
            ${CMAKE_COMMAND}
            "-DEXAMPLE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/examples/particle_feedback"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_public_api.cmake"
    )
    set_tests_properties(example.public_api PROPERTIES LABELS "example;api")

    # Binary-level contract for the example CLI. This complements, rather than
    # replaces, the in-process Catch2 matrix: every case runs the real
    # executable and checks its exit status and histogram sum.
    set(GRAPHFLOW_EXAMPLE_CLI_PARTICLES 0 1 17 127 128 4097)
    set(GRAPHFLOW_EXAMPLE_CLI_CHUNKS 1 31 127 128)
    set(GRAPHFLOW_EXAMPLE_CLI_SEED 42)
    set(GRAPHFLOW_EXAMPLE_CLI_MAX_VISITS 64)
    set(GRAPHFLOW_EXAMPLE_CLI_TIMEOUT 300)

    if(TARGET graphFlow_particle_feedback)
        set(
            GRAPHFLOW_EXAMPLE_CLI_BINARY
            "$<TARGET_FILE:graphFlow_particle_feedback>"
        )
    else()
        # Keep the CPU contract active even when examples are disabled: the
        # driver reports the missing binary and the CTest case fails.
        set(
            GRAPHFLOW_EXAMPLE_CLI_BINARY
            "${CMAKE_CURRENT_BINARY_DIR}/graphFlow_particle_feedback${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endif()

    function(graphflow_add_example_cli_test backend particles chunk_size)
        set(test_name "example.particle_feedback.${backend}.N${particles}.C${chunk_size}")
        add_test(
            NAME ${test_name}
            COMMAND
                ${CMAKE_COMMAND}
                "-DBINARY=${GRAPHFLOW_EXAMPLE_CLI_BINARY}"
                "-DBACKEND=${backend}"
                "-DPARTICLES=${particles}"
                "-DCHUNK_SIZE=${chunk_size}"
                "-DSEED=${GRAPHFLOW_EXAMPLE_CLI_SEED}"
                "-DMAX_VISITS=${GRAPHFLOW_EXAMPLE_CLI_MAX_VISITS}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_cli.cmake"
        )
        set_tests_properties(
            ${test_name}
            PROPERTIES
                TIMEOUT ${GRAPHFLOW_EXAMPLE_CLI_TIMEOUT}
                LABELS "example;cli;${backend}"
        )
    endfunction()

    # The CPU backend is always built. A CUDA build adds the CUDA cases; a CUDA
    # case fails when no usable CUDA device is visible instead of being skipped.
    foreach(particles IN LISTS GRAPHFLOW_EXAMPLE_CLI_PARTICLES)
        foreach(chunk_size IN LISTS GRAPHFLOW_EXAMPLE_CLI_CHUNKS)
            graphflow_add_example_cli_test(cpu ${particles} ${chunk_size})
            if(GRAPHFLOW_ENABLE_CUDA)
                graphflow_add_example_cli_test(cuda ${particles} ${chunk_size})
            endif()
        endforeach()
    endforeach()

    add_test(
        NAME example.photon_public_api
        COMMAND
            ${CMAKE_COMMAND}
            "-DEXAMPLE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/examples/photon_scattering"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_photon_public_api.cmake"
    )
    set_tests_properties(example.photon_public_api PROPERTIES LABELS "example;api;photon")

    if(TARGET graphFlow_photon_scattering)
        set(
            GRAPHFLOW_PHOTON_CLI_BINARY
            "$<TARGET_FILE:graphFlow_photon_scattering>"
        )
    else()
        set(
            GRAPHFLOW_PHOTON_CLI_BINARY
            "${CMAKE_CURRENT_BINARY_DIR}/graphFlow_photon_scattering${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endif()

    function(graphflow_add_photon_cli_test backend photons chunk_size mode sigma_s sigma_a cap seed)
        set(
            test_name
            "example.photon.${backend}.N${photons}.C${chunk_size}.${mode}.S${sigma_s}.A${sigma_a}.K${cap}"
        )
        add_test(
            NAME ${test_name}
            COMMAND
                ${CMAKE_COMMAND}
                "-DBINARY=${GRAPHFLOW_PHOTON_CLI_BINARY}"
                "-DBACKEND=${backend}"
                "-DPHOTONS=${photons}"
                "-DCHUNK_SIZE=${chunk_size}"
                "-DSEED=${seed}"
                "-DSIGMA_S=${sigma_s}"
                "-DSIGMA_A=${sigma_a}"
                "-DMODE=${mode}"
                "-DCAP=${cap}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_photon_cli.cmake"
        )
        set_tests_properties(
            ${test_name}
            PROPERTIES
                TIMEOUT 300
                LABELS "example;cli;photon;${backend}"
        )
    endfunction()

    foreach(photons IN ITEMS 0 1 127 4097)
        foreach(chunk_size IN ITEMS 1 127)
            graphflow_add_photon_cli_test(cpu ${photons} ${chunk_size} forward 1.0 0.0 64 42)
            if(GRAPHFLOW_ENABLE_CUDA)
                graphflow_add_photon_cli_test(cuda ${photons} ${chunk_size} forward 1.0 0.0 64 42)
            endif()
        endforeach()
    endforeach()
    graphflow_add_photon_cli_test(cpu 4097 31 isotropic 0.7 0.3 256 7)
    graphflow_add_photon_cli_test(cpu 512 31 forward 1.0 0.0 2 42)
    if(GRAPHFLOW_ENABLE_CUDA)
        graphflow_add_photon_cli_test(cuda 4097 31 isotropic 0.7 0.3 256 7)
        graphflow_add_photon_cli_test(cuda 512 31 forward 1.0 0.0 2 42)
    endif()
endif()
