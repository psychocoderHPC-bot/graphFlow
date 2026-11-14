# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Binary-level contract test for the graphFlow_photon_scattering example.
#
# The driver runs the real executable, verifies the selected backend and the
# requested photon count, and checks that terminal event counts sum to N.

foreach(required BINARY BACKEND PHOTONS CHUNK_SIZE SEED SIGMA_S SIGMA_A MODE CAP)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "example_photon_cli.cmake: missing -D${required}=...")
    endif()
endforeach()

if(NOT EXISTS "${BINARY}")
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: example binary '${BINARY}' does not exist. "
        "Build the graphFlow_photon_scattering target (GRAPHFLOW_BUILD_EXAMPLES=ON) before running CTest."
    )
endif()

set(command
    "${BINARY}"
    --backend
    "${BACKEND}"
    --photons
    "${PHOTONS}"
    --thickness
    "1"
    --sigma-s
    "${SIGMA_S}"
    --sigma-a
    "${SIGMA_A}"
    --mode
    "${MODE}"
    --max-scatters
    "${CAP}"
    --chunk-size
    "${CHUNK_SIZE}"
    --seed
    "${SEED}"
)

execute_process(
    COMMAND ${command}
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_stdout
    ERROR_VARIABLE run_stderr
    TIMEOUT 300
)

if(NOT "${run_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: '${BINARY}' failed for backend=${BACKEND} N=${PHOTONS} C=${CHUNK_SIZE}.\n"
        "exit: ${run_result}\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

set(expected_run_line "# graphFlow_photon_scattering backend=${BACKEND} device=0 ")
string(FIND "${run_stdout}" "${expected_run_line}" run_line_position)
if(run_line_position EQUAL -1)
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: no photon example header for backend '${BACKEND}'.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

string(FIND "${run_stdout}" " photons=${PHOTONS} " photon_position)
if(photon_position EQUAL -1)
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: output does not report photons=${PHOTONS}.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()

string(REPLACE "\r" "" run_stdout "${run_stdout}")
string(REPLACE "\n" ";" output_lines "${run_stdout}")

set(in_events FALSE)
set(saw_events FALSE)
set(count_sum 0)
set(terminal_sum -1)

foreach(line IN LISTS output_lines)
    if(line MATCHES "^# terminal-sum,([0-9]+)")
        math(EXPR terminal_sum "${CMAKE_MATCH_1}")
        continue()
    endif()
    if(line STREQUAL "# event,count")
        set(in_events TRUE)
        set(saw_events TRUE)
        continue()
    endif()
    if(NOT in_events)
        continue()
    endif()
    if(line STREQUAL "# outcome,scatter-count,count")
        set(in_events FALSE)
        continue()
    endif()
    if(line STREQUAL "" OR line MATCHES "^#")
        continue()
    endif()
    if(NOT line MATCHES "^(transmitted|reflected|absorbed|capped),([0-9]+)$")
        message(
            FATAL_ERROR
            "example_photon_cli.cmake: malformed event row '${line}' for backend=${BACKEND} N=${PHOTONS}."
        )
    endif()
    math(EXPR count_sum "${count_sum} + ${CMAKE_MATCH_2}")
endforeach()

if(NOT saw_events)
    message(FATAL_ERROR "example_photon_cli.cmake: no '# event,count' header.")
endif()
if(NOT count_sum EQUAL PHOTONS)
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: terminal event counts sum to ${count_sum}, expected ${PHOTONS} "
        "for backend=${BACKEND} C=${CHUNK_SIZE}.\n"
        "stdout:\n${run_stdout}\n"
        "stderr:\n${run_stderr}"
    )
endif()
if(NOT terminal_sum EQUAL PHOTONS)
    message(
        FATAL_ERROR
        "example_photon_cli.cmake: '# terminal-sum' is ${terminal_sum}, expected ${PHOTONS}."
    )
endif()

message(
    STATUS
    "example_photon_cli: backend=${BACKEND} N=${PHOTONS} C=${CHUNK_SIZE} mode=${MODE} "
    "exit=0 terminal-sum=${terminal_sum}"
)
