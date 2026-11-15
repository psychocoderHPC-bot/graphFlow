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

set(GRAPHFLOW_GIT_COMMIT "unknown" CACHE STRING "graphFlow commit written to photon metadata")
set(GRAPHFLOW_ALPAKA_PIN "unknown" CACHE STRING "alpaka commit written to photon metadata")
if(GRAPHFLOW_GIT_COMMIT STREQUAL "unknown")
    execute_process(
        COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE _graphflow_git_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_graphflow_git_commit)
        set(GRAPHFLOW_GIT_COMMIT "${_graphflow_git_commit}" CACHE STRING "graphFlow commit written to photon metadata" FORCE)
    endif()
endif()

option(GRAPHFLOW_ENABLE_CUDA "Compile the CUDA backend (sets alpaka_DEP_CUDA=ON)" OFF)
option(
    GRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS
    "Register the opt-in large photon diagnostics CTest cases"
    OFF
)
option(
    GRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK
    "Register the error-path CTest case that fails when LeakSanitizer reports a leak"
    OFF
)
option(GRAPHFLOW_ENABLE_HIP "Compile the HIP backend (sets alpaka_DEP_HIP=ON)" OFF)
option(
    GRAPHFLOW_ENABLE_SYCL
    "Compile the OneAPI/SYCL backend for Intel GPUs (requires icpx; sets alpaka_DEP_ONEAPI=ON)"
    OFF
)
# Default the debug validation to the build type: ON for CMAKE_BUILD_TYPE=Debug
# and OFF otherwise. An explicit
# -DGRAPHFLOW_ENABLE_DEBUG_VALIDATION=ON|OFF overrides the default because
# option() keeps an existing cache entry.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_graphflow_debug_validation_default ON)
else()
    set(_graphflow_debug_validation_default OFF)
endif()
option(
    GRAPHFLOW_ENABLE_DEBUG_VALIDATION
    "Compile the debug-only per-event payload validation"
    ${_graphflow_debug_validation_default}
)
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

if(GRAPHFLOW_ALPAKA_PIN STREQUAL "unknown")
    execute_process(
        COMMAND git -C "${GRAPHFLOW_ALPAKA_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE _graphflow_alpaka_pin
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_graphflow_alpaka_pin)
        set(GRAPHFLOW_ALPAKA_PIN "${_graphflow_alpaka_pin}" CACHE STRING "alpaka commit written to photon metadata" FORCE)
    endif()
endif()

set(_graphflow_accelerator_count 0)
foreach(_graphflow_backend_option GRAPHFLOW_ENABLE_CUDA GRAPHFLOW_ENABLE_HIP GRAPHFLOW_ENABLE_SYCL)
    if(${_graphflow_backend_option})
        math(EXPR _graphflow_accelerator_count "${_graphflow_accelerator_count} + 1")
    endif()
endforeach()
if(_graphflow_accelerator_count GREATER 1)
    message(
        FATAL_ERROR
        "GRAPHFLOW_ENABLE_CUDA, GRAPHFLOW_ENABLE_HIP and GRAPHFLOW_ENABLE_SYCL are mutually exclusive in one "
        "build; use one build directory per backend"
    )
endif()

# Accelerator languages and the matching alpaka dependency options must be established before
# add_subdirectory(alpaka) so alpaka builds its CUDA/HIP/SYCL targets and alpaka_finalize() compiles
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

if(GRAPHFLOW_ENABLE_SYCL)
    get_filename_component(_graphflow_cxx_compiler_name "${CMAKE_CXX_COMPILER}" NAME)
    if(NOT _graphflow_cxx_compiler_name MATCHES "^icpx$")
        message(
            FATAL_ERROR
            "GRAPHFLOW_ENABLE_SYCL=ON requires the Intel oneAPI DPC++ compiler (icpx), but "
            "CMAKE_CXX_COMPILER is '${CMAKE_CXX_COMPILER}'. Source /opt/intel/oneapi/setvars.sh and "
            "configure with -DCMAKE_CXX_COMPILER=icpx."
        )
    endif()
    # The pinned alpaka 3.0.0 has two malformed calls in cmake/alpakaOneApi.cmake:
    # - the relocatable-device-code branch omits the 'target' argument type, so any explicit
    #   ON/OFF value aborts the configure in alpaka_set_compiler_options();
    # - the "-fsycl-targets" call passes a stray INTERFACE argument, which CMake currently
    #   absorbs as a repeated scope keyword but which graphFlow sanitizes after the include.
    # graphFlow does not use relocatable device code, so keep the upstream DEFAULT and document
    # the pin limitation instead of patching the external alpaka checkout.
    if(DEFINED alpaka_RELOCATABLE_DEVICE_CODE AND NOT alpaka_RELOCATABLE_DEVICE_CODE STREQUAL "DEFAULT")
        message(
            STATUS
            "graphFlow: forcing alpaka_RELOCATABLE_DEVICE_CODE=DEFAULT for the pinned alpaka 3.0.0 SYCL build"
        )
    endif()
    set(
        alpaka_RELOCATABLE_DEVICE_CODE
        "DEFAULT"
        CACHE STRING "Relocatable device code (pinned alpaka 3.0.0 SYCL workaround)" FORCE
    )
    set(alpaka_DEP_ONEAPI ON CACHE BOOL "Enable the SYCL dependency in alpaka" FORCE)
    set(alpaka_ONEAPI_IntelGpu ON CACHE BOOL "Enable api::oneApi/deviceKind::intelGpu in alpaka" FORCE)
    set(alpaka_ONEAPI_Cpu OFF CACHE BOOL "Disable the SYCL CPU device in alpaka" FORCE)
    set(alpaka_ONEAPI_NvidiaGpu OFF CACHE BOOL "Disable the SYCL NVIDIA device in alpaka" FORCE)
    set(alpaka_ONEAPI_AmdGpu OFF CACHE BOOL "Disable the SYCL AMD device in alpaka" FORCE)
endif()

# alpaka3 is consumed from the pinned source tree; the project only uses its public API.
add_subdirectory("${GRAPHFLOW_ALPAKA_SOURCE_DIR}" alpaka)

if(GRAPHFLOW_ENABLE_SYCL)
    # Sanitize the pinned alpaka 3.0.0 stray INTERFACE argument: if the property ever contains a
    # literal INTERFACE token, icpx would reject it as an unknown argument.
    get_target_property(_graphflow_oneapi_compile_options alpaka_target_oneapi INTERFACE_COMPILE_OPTIONS)
    if(_graphflow_oneapi_compile_options AND NOT _graphflow_oneapi_compile_options STREQUAL "_graphflow_oneapi_compile_options-NOTFOUND")
        list(REMOVE_ITEM _graphflow_oneapi_compile_options INTERFACE)
        set_target_properties(
            alpaka_target_oneapi
            PROPERTIES INTERFACE_COMPILE_OPTIONS "${_graphflow_oneapi_compile_options}"
        )
    endif()
endif()

add_library(graphFlow INTERFACE)
add_library(graphFlow::graphFlow ALIAS graphFlow)
target_compile_features(graphFlow INTERFACE cxx_std_20)
target_include_directories(
    graphFlow
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(graphFlow INTERFACE alpaka::alpaka)
if(GRAPHFLOW_ENABLE_SYCL)
    # The SYCL build compiles oneAPI kernels; link the dependency target explicitly even though
    # alpaka::alpaka also pulls it in when ONEAPI is the only enabled API dependency.
    target_link_libraries(graphFlow INTERFACE alpaka::oneapi)
endif()

if(GRAPHFLOW_ENABLE_DEBUG_VALIDATION)
    target_compile_definitions(graphFlow INTERFACE GRAPHFLOW_DEBUG_VALIDATION=1)
endif()

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
    if(GRAPHFLOW_ENABLE_SYCL)
        target_sources(
            graphFlow_photon_scattering
            PRIVATE examples/photon_scattering/PhotonScatteringFp64OneApi.cpp
        )
    endif()
    target_link_libraries(graphFlow_photon_scattering PRIVATE graphFlow::graphFlow)
    target_include_directories(
        graphFlow_photon_scattering
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/photon_scattering
    )
    target_compile_definitions(
        graphFlow_photon_scattering
        PRIVATE
            GRAPHFLOW_GIT_COMMIT="${GRAPHFLOW_GIT_COMMIT}"
            GRAPHFLOW_ALPAKA_PIN="${GRAPHFLOW_ALPAKA_PIN}"
            GRAPHFLOW_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            GRAPHFLOW_ALPAKA_SOURCE_DIR="${GRAPHFLOW_ALPAKA_SOURCE_DIR}"
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
        tests/integration/test_kernel_axes.cpp
        tests/integration/test_particle_feedback.cpp
        tests/integration/test_scripted_routing.cpp
        tests/integration/test_pressure.cpp
        tests/integration/test_admission.cpp
        tests/integration/test_failure_paths.cpp
        tests/integration/test_batching.cpp
        tests/integration/test_async_execution.cpp
        tests/integration/test_scheduler_validation.cpp
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
                tests/integration/test_failure_paths_cuda.cpp
                tests/integration/test_photon_scattering_cuda.cpp
        )
    endif()
    if(GRAPHFLOW_ENABLE_HIP)
        # Runtime cases: they fail when no AMD device is visible. GitHub CI builds
        # HIP compile-only because its runners have no AMD GPU.
        target_sources(
            graphFlow_tests
            PRIVATE
                tests/integration/test_flow_execution_hip.cpp
                tests/integration/test_failure_paths_hip.cpp
        )
    endif()
    if(GRAPHFLOW_ENABLE_SYCL)
        # Runtime coverage: every case fails when no Intel GPU is visible. Hosted CI compiles this
        # target in the sycl-compile job but runs no CTest; the full device matrix runs on hal with
        # an Arc A770.
        target_sources(
            graphFlow_tests
            PRIVATE
                tests/unit/test_sycl_kernel.cpp
                tests/integration/test_flow_execution_sycl.cpp
                tests/integration/test_failure_paths_sycl.cpp
                tests/integration/test_photon_scattering_sycl.cpp
                tests/integration/test_async_execution_sycl.cpp
                tests/integration/test_framespec_sycl.cpp
                tests/integration/test_particle_pressure_sycl.cpp
                tests/integration/test_photon_scattering_sycl_fp64.cpp
                examples/photon_scattering/PhotonScatteringFp64OneApi.cpp
        )
    endif()
    # icpx 2026.2 miscompiles a 32-byte double functor passed as a kernel argument under
    # Intel FP64 emulation at -O1 and above: the transform writes an invalid payload and the
    # device can be lost. Keep only the double oneAPI photon translation unit unoptimized; the
    # fp32 path and the rest of the suite stay optimized.
    set_source_files_properties(
        examples/photon_scattering/PhotonScatteringFp64OneApi.cpp
        PROPERTIES COMPILE_OPTIONS "-O0"
    )
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
    # Large statistical photon runs are opt-in. The default suite keeps only fast cases.
    # SYCL cases are discovered under their own label so `ctest -L sycl` selects the whole
    # device matrix; the other specs exclude that tag to keep every case registered once.
    catch_discover_tests(graphFlow_tests TEST_SPEC "~[diagnostics]~[sycl]~[sycl-fp64]")
    if(GRAPHFLOW_ENABLE_PHOTON_DIAGNOSTICS)
        catch_discover_tests(
            graphFlow_tests
            TEST_SPEC "[diagnostics]~[sycl]"
            TEST_PREFIX "diagnostics."
            PROPERTIES LABELS "diagnostics"
        )
    endif()
    if(GRAPHFLOW_ENABLE_SYCL)
        catch_discover_tests(
            graphFlow_tests
            TEST_SPEC "[sycl]"
            PROPERTIES LABELS "sycl"
        )
        # Explicit Intel FP64 emulation is environment-scoped: the IGC variables are read by the
        # SYCL runtime before the first queue exists, so they can only be supplied to a fresh
        # process. `cmake -E env` gives this one Catch2 case the three variables without touching
        # the other SYCL tests. The case is kept out of the default and [sycl] discovery specs and
        # runs with a hard timeout for the known DG2 hang/DEVICE_LOST failure mode (upstream issue
        # #1036).
        add_test(
            NAME test_photon_scattering_sycl_fp64
            COMMAND
                ${CMAKE_COMMAND} -E env
                GRAPHFLOW_SYCL_FP64_EMULATION=1
                IGC_EnableDPEmulation=1
                OverrideDefaultFP64Settings=1
                $<TARGET_FILE:graphFlow_tests>
                "SYCL photon FP64 emulation runs the exact double path"
        )
        set_tests_properties(
            test_photon_scattering_sycl_fp64
            PROPERTIES LABELS "sycl;fp64;emulation" TIMEOUT 300
        )
    endif()

    # Error-path leak check: run every [failure] and [status] case in one process.
    # LeakSanitizer, enabled by the ASAN build, fails the process at exit if a
    # failure path leaks. The sanitizer CI job turns this on; the default suite
    # is unchanged.
    if(GRAPHFLOW_ENABLE_ERROR_PATH_LEAK_CHECK)
        add_test(NAME graphflow.error_path_leak_check COMMAND graphFlow_tests "[failure],[status]")
        set_tests_properties(
            graphflow.error_path_leak_check
            PROPERTIES
                LABELS "sanitizer;leak;error-path"
                ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;LSAN_OPTIONS=detect_leaks=1:exitcode=66"
                TIMEOUT 900
        )
    endif()

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
    # The N=1e6 smoke runs inside the ordinary CTest job. The driver gets a shorter
    # internal timeout so a hang is reported by the script before CTest kills it.
    set(GRAPHFLOW_EXAMPLE_SMOKE_TIMEOUT 960)
    set(GRAPHFLOW_EXAMPLE_SMOKE_DRIVER_TIMEOUT 900)

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
            if(GRAPHFLOW_ENABLE_HIP)
                graphflow_add_example_cli_test(hip ${particles} ${chunk_size})
            endif()
            if(GRAPHFLOW_ENABLE_SYCL)
                graphflow_add_example_cli_test(sycl ${particles} ${chunk_size})
            endif()
        endforeach()
    endforeach()

    # Scale smoke: the fixed-seed N=1e6 CPU run must stay inside a finite timeout
    # and produce an exact histogram. It runs unconditionally in the normal CTest
    # job, so hosted CPU CI exercises it without a skip. CUDA and HIP remain
    # compile-only there; their runtime smoke stays a manual rosi/hal check.
    add_test(
        NAME example.particle_feedback.smoke.N1000000
        COMMAND
            ${CMAKE_COMMAND}
            "-DBINARY=${GRAPHFLOW_EXAMPLE_CLI_BINARY}"
            "-DBACKEND=cpu"
            "-DPARTICLES=1000000"
            "-DCHUNK_SIZE=512"
            "-DSEED=${GRAPHFLOW_EXAMPLE_CLI_SEED}"
            "-DMAX_VISITS=${GRAPHFLOW_EXAMPLE_CLI_MAX_VISITS}"
            "-DTIMEOUT=${GRAPHFLOW_EXAMPLE_SMOKE_DRIVER_TIMEOUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_cli.cmake"
    )
    set_tests_properties(
        example.particle_feedback.smoke.N1000000
        PROPERTIES
            TIMEOUT ${GRAPHFLOW_EXAMPLE_SMOKE_TIMEOUT}
            LABELS "example;cli;cpu;smoke;long"
    )

    # Subprocess failure contract: invalid CLI input and one runtime-failing
    # example configuration must exit nonzero through the real executable.
    add_test(
        NAME example.particle_feedback.failure_paths
        COMMAND
            ${CMAKE_COMMAND}
            "-DBINARY=${GRAPHFLOW_EXAMPLE_CLI_BINARY}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_cli_failure.cmake"
    )
    set_tests_properties(
        example.particle_feedback.failure_paths
        PROPERTIES
            TIMEOUT 300
            LABELS "example;cli;failure"
    )

    # Cross-backend exact-equality cases. Every backend compiled into this build
    # runs the same seed/N/C/P/Q and the driver compares the sorted histogram rows
    # and the count sum exactly. A CPU-only build checks the CPU run against
    # itself; a CUDA or HIP build adds the device backend. A compiled backend with
    # no usable device fails the case instead of being skipped. The GPU cases are
    # registered only for compiled backends, so a compile-only hosted job makes no
    # runtime claim.
    set(GRAPHFLOW_CROSS_BACKEND_BACKENDS cpu)
    if(GRAPHFLOW_ENABLE_CUDA)
        list(APPEND GRAPHFLOW_CROSS_BACKEND_BACKENDS cuda)
    endif()
    if(GRAPHFLOW_ENABLE_HIP)
        list(APPEND GRAPHFLOW_CROSS_BACKEND_BACKENDS hip)
    endif()
    if(GRAPHFLOW_ENABLE_SYCL)
        list(APPEND GRAPHFLOW_CROSS_BACKEND_BACKENDS sycl)
    endif()
    set(GRAPHFLOW_CROSS_BACKEND_LABELS "example;cli;cross-backend;long")
    foreach(backend IN LISTS GRAPHFLOW_CROSS_BACKEND_BACKENDS)
        list(APPEND GRAPHFLOW_CROSS_BACKEND_LABELS "${backend}")
    endforeach()
    # Comma-separated so the backend list survives add_test's argument handling.
    string(REPLACE ";" "," GRAPHFLOW_CROSS_BACKEND_BACKENDS_CSV "${GRAPHFLOW_CROSS_BACKEND_BACKENDS}")
    set(GRAPHFLOW_CROSS_BACKEND_PARTICLES 1000000)
    set(GRAPHFLOW_CROSS_BACKEND_CHUNK_SIZE 128)
    set(GRAPHFLOW_CROSS_BACKEND_LIVE_PARTICLES 1024)
    set(GRAPHFLOW_CROSS_BACKEND_RUN_TIMEOUT 900)
    set(GRAPHFLOW_CROSS_BACKEND_TEST_TIMEOUT 1900)

    function(graphflow_add_cross_backend_test batch_chunks)
        set(test_name "example.particle_feedback.cross_backend.Q${batch_chunks}")
        add_test(
            NAME ${test_name}
            COMMAND
                ${CMAKE_COMMAND}
                "-DBINARY=${GRAPHFLOW_EXAMPLE_CLI_BINARY}"
                "-DBACKENDS=${GRAPHFLOW_CROSS_BACKEND_BACKENDS_CSV}"
                "-DPARTICLES=${GRAPHFLOW_CROSS_BACKEND_PARTICLES}"
                "-DCHUNK_SIZE=${GRAPHFLOW_CROSS_BACKEND_CHUNK_SIZE}"
                "-DLIVE_PARTICLES=${GRAPHFLOW_CROSS_BACKEND_LIVE_PARTICLES}"
                "-DBATCH_CHUNKS=${batch_chunks}"
                "-DSEED=${GRAPHFLOW_EXAMPLE_CLI_SEED}"
                "-DMAX_VISITS=${GRAPHFLOW_EXAMPLE_CLI_MAX_VISITS}"
                "-DTIMEOUT=${GRAPHFLOW_CROSS_BACKEND_RUN_TIMEOUT}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_cross_backend.cmake"
        )
        set_tests_properties(
            ${test_name}
            PROPERTIES
                TIMEOUT ${GRAPHFLOW_CROSS_BACKEND_TEST_TIMEOUT}
                LABELS "${GRAPHFLOW_CROSS_BACKEND_LABELS}"
        )
    endfunction()

    foreach(batch_chunks IN ITEMS 1 2 8)
        graphflow_add_cross_backend_test(${batch_chunks})
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
            if(GRAPHFLOW_ENABLE_HIP)
                graphflow_add_photon_cli_test(hip ${photons} ${chunk_size} forward 1.0 0.0 64 42)
            endif()
            if(GRAPHFLOW_ENABLE_SYCL)
                graphflow_add_photon_cli_test(sycl ${photons} ${chunk_size} forward 1.0 0.0 64 42)
            endif()
        endforeach()
    endforeach()
    graphflow_add_photon_cli_test(cpu 4097 31 isotropic 0.7 0.3 256 7)
    graphflow_add_photon_cli_test(cpu 512 31 forward 1.0 0.0 2 42)
    if(GRAPHFLOW_ENABLE_CUDA)
        graphflow_add_photon_cli_test(cuda 4097 31 isotropic 0.7 0.3 256 7)
        graphflow_add_photon_cli_test(cuda 512 31 forward 1.0 0.0 2 42)
    endif()
    if(GRAPHFLOW_ENABLE_HIP)
        graphflow_add_photon_cli_test(hip 4097 31 isotropic 0.7 0.3 256 7)
        graphflow_add_photon_cli_test(hip 512 31 forward 1.0 0.0 2 42)
    endif()
    if(GRAPHFLOW_ENABLE_SYCL)
        graphflow_add_photon_cli_test(sycl 4097 31 isotropic 0.7 0.3 256 7)
        graphflow_add_photon_cli_test(sycl 512 31 forward 1.0 0.0 2 42)
    endif()

    function(graphflow_add_photon_output_test backend)
        set(test_name "example.photon.output.${backend}")
        add_test(
            NAME ${test_name}
            COMMAND
                ${CMAKE_COMMAND}
                "-DBINARY=${GRAPHFLOW_PHOTON_CLI_BINARY}"
                "-DBACKEND=${backend}"
                "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/photon-output-${backend}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/example_photon_output.cmake"
        )
        set_tests_properties(
            ${test_name}
            PROPERTIES
                TIMEOUT 300
                LABELS "example;photon;output;${backend}"
        )
    endfunction()

    graphflow_add_photon_output_test(cpu)
    if(GRAPHFLOW_ENABLE_CUDA)
        graphflow_add_photon_output_test(cuda)
    endif()
    if(GRAPHFLOW_ENABLE_HIP)
        graphflow_add_photon_output_test(hip)
    endif()
    if(GRAPHFLOW_ENABLE_SYCL)
        graphflow_add_photon_output_test(sycl)
    endif()
endif()
