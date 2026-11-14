# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Binary-level failure contract for the graphFlow_particle_feedback example.
#
# The driver runs the real executable with inputs that the documented CLI
# contract rejects and with a run configuration that fails inside the example.
# Every case must exit nonzero; a zero exit is a test failure. This complements
# the in-process Catch2 CLI tests, which never cross a process boundary.

if(NOT DEFINED BINARY)
    message(FATAL_ERROR "example_cli_failure.cmake: missing -DBINARY=...")
endif()

if(NOT EXISTS "${BINARY}")
    message(
        FATAL_ERROR
        "example_cli_failure.cmake: example binary '${BINARY}' does not exist. "
        "Build the graphFlow_particle_feedback target (GRAPHFLOW_BUILD_EXAMPLES=ON) before running CTest."
    )
endif()

# Run one failing invocation and require a nonzero exit plus the expected stderr
# fragment. `expected_exit` pins the documented code where the example defines one.
function(graphflow_expect_failure name expected_exit expected_message)
    execute_process(
        COMMAND "${BINARY}" ${ARGN}
        RESULT_VARIABLE run_result
        OUTPUT_VARIABLE run_stdout
        ERROR_VARIABLE run_stderr
        TIMEOUT 120
    )
    if(NOT "${run_result}" MATCHES "^-?[0-9]+$")
        message(
            FATAL_ERROR
            "example_cli_failure.cmake: case '${name}' did not exit normally (result='${run_result}').\n"
            "stdout:\n${run_stdout}\nstderr:\n${run_stderr}"
        )
    endif()
    if(run_result EQUAL 0)
        message(
            FATAL_ERROR
            "example_cli_failure.cmake: case '${name}' exited 0; a failure path must return nonzero.\n"
            "stdout:\n${run_stdout}\nstderr:\n${run_stderr}"
        )
    endif()
    if(NOT run_result EQUAL expected_exit)
        message(
            FATAL_ERROR
            "example_cli_failure.cmake: case '${name}' exited ${run_result}, expected ${expected_exit}.\n"
            "stdout:\n${run_stdout}\nstderr:\n${run_stderr}"
        )
    endif()
    string(FIND "${run_stderr}" "${expected_message}" message_position)
    if(message_position EQUAL -1)
        message(
            FATAL_ERROR
            "example_cli_failure.cmake: case '${name}' stderr misses '${expected_message}'.\n"
            "stderr:\n${run_stderr}"
        )
    endif()
    message(STATUS "example_cli_failure: case '${name}' exit=${run_result}")
endfunction()

# Parse-time invalid inputs: the example rejects them with exit 2 before it
# selects or touches a device, so these cases run on every backend build.
graphflow_expect_failure(unknown-backend 2 "unknown backend 'tpu'" --backend tpu)
graphflow_expect_failure(unknown-flag 2 "unknown argument '--not-a-flag'" --not-a-flag)
graphflow_expect_failure(invalid-number 2 "--particles: invalid value '1x'" --particles 1x)
graphflow_expect_failure(zero-chunk 2 "--chunk-size: value must be non-zero" --chunk-size 0)
graphflow_expect_failure(zero-live-particles 2 "--live-particles: value must be non-zero" --live-particles 0)
graphflow_expect_failure(zero-tile 2 "--tile-size: value must be non-zero" --tile-size 0)
graphflow_expect_failure(zero-batch 2 "--batch-chunks: value must be non-zero" --batch-chunks 0)
graphflow_expect_failure(zero-frame-cap 2 "--frame-cap: value must be non-zero" --frame-cap 0)

# Runtime failing run path: the example rejects a max-visits value above the
# RNG stream-event range after device selection and returns exit 1. This crosses
# the process boundary through the real example run, not just the parser.
graphflow_expect_failure(
    max-visits-overflow
    1
    "max visits exceeds the RNG stream-event range"
    --backend
    cpu
    --particles
    1
    --max-visits
    65536
)

message(STATUS "example_cli_failure: all failure cases returned nonzero")
