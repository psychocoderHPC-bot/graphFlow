# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Binary-level contract test for the graphFlow_particle_feedback example.
#
# This driver is intentionally independent of the Catch2 suite: it runs the
# installed example binary, checks that the requested backend really ran, and
# verifies that the sorted weight histogram counts sum to the requested N.
# Missing binaries and unavailable backends are fatal, so a required CTest case
# fails instead of silently passing.

foreach(required BINARY BACKEND PARTICLES CHUNK_SIZE SEED MAX_VISITS)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "example_cli.cmake: missing -D${required}=...")
    endif()
endforeach()

if(NOT EXISTS "${BINARY}")
    message(
        FATAL_ERROR
        "example_cli.cmake: example binary '${BINARY}' does not exist. "
        "Build the graphFlow_particle_feedback target (GRAPHFLOW_BUILD_EXAMPLES=ON) before running CTest."
    )
endif()

set(command
    "${BINARY}"
    --backend
    "${BACKEND}"
    --particles
    "${PARTICLES}"
    --chunk-size
    "${CHUNK_SIZE}"
    --seed
    "${SEED}"
    --max-visits
    "${MAX_VISITS}"
)

execute_process(
    COMMAND ${command}
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_stdout
    ERROR_VARIABLE run_stderr
    TIMEOUT 240
)

if(NOT "${run_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "example_cli.cmake: '${BINARY}' failed for backend=${BACKEND} N=${PARTICLES} C=${CHUNK_SIZE} "
        "seed=${SEED} max-visits=${MAX_VISITS}.\n"
        "exit: ${run_result}\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

# The binary reports the backend and particle count it actually executed. This
# rejects a silent fallback to a different backend.
set(expected_run_line
    "# graphFlow_particle_feedback backend=${BACKEND} device=0 "
)
string(FIND "${run_stdout}" "${expected_run_line}" run_line_position)
if(run_line_position EQUAL -1)
    message(
        FATAL_ERROR
        "example_cli.cmake: no example header for backend '${BACKEND}' in the output.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

string(FIND "${run_stdout}" " particles=${PARTICLES} " particle_position)
if(particle_position EQUAL -1)
    message(
        FATAL_ERROR
        "example_cli.cmake: output does not report particles=${PARTICLES}.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

# Parse the histogram rows after the "# weight,count" header. Comment lines
# (including the optional non-terminating count) are ignored.
string(REPLACE "\r" "" run_stdout "${run_stdout}")
string(REPLACE "\n" ";" output_lines "${run_stdout}")

set(in_histogram FALSE)
set(saw_histogram FALSE)
set(count_sum 0)
set(row_count 0)

foreach(line IN LISTS output_lines)
    if(line STREQUAL "# weight,count")
        set(in_histogram TRUE)
        set(saw_histogram TRUE)
        continue()
    endif()
    if(NOT in_histogram OR line STREQUAL "" OR line MATCHES "^#")
        continue()
    endif()
    if(NOT line MATCHES "^([0-9]+),([0-9]+)$")
        message(
            FATAL_ERROR
            "example_cli.cmake: malformed histogram row '${line}' for backend=${BACKEND} N=${PARTICLES} C=${CHUNK_SIZE}."
        )
    endif()
    math(EXPR count_sum "${count_sum} + ${CMAKE_MATCH_2}")
    math(EXPR row_count "${row_count} + 1")
endforeach()

if(NOT saw_histogram)
    message(
        FATAL_ERROR
        "example_cli.cmake: no '# weight,count' header for backend=${BACKEND} N=${PARTICLES} C=${CHUNK_SIZE}.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

if(NOT count_sum EQUAL PARTICLES)
    message(
        FATAL_ERROR
        "example_cli.cmake: histogram counts sum to ${count_sum}, expected ${PARTICLES} "
        "for backend=${BACKEND} C=${CHUNK_SIZE} seed=${SEED} max-visits=${MAX_VISITS}.\n"
        "histogram rows: ${row_count}\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

message(
    STATUS
    "example_cli: backend=${BACKEND} N=${PARTICLES} C=${CHUNK_SIZE} seed=${SEED} "
    "max-visits=${MAX_VISITS} exit=0 histogram-rows=${row_count} count-sum=${count_sum}"
)
