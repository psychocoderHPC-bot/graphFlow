# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Checks the machine-readable photon output directory and bounded trace files.

foreach(required BINARY BACKEND OUTPUT_DIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "example_photon_output.cmake: missing -D${required}=...")
    endif()
endforeach()

if(NOT EXISTS "${BINARY}")
    message(FATAL_ERROR "example_photon_output.cmake: missing binary '${BINARY}'")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")

set(command
    "${BINARY}"
    --backend
    "${BACKEND}"
    --photons
    "257"
    --thickness
    "1"
    --sigma-s
    "1"
    --sigma-a
    "0.2"
    --mode
    "isotropic"
    --seed
    "42"
    --chunk-size
    "31"
    --trace-photons
    "8"
    --trace-vertices
    "64"
    --output-dir
    "${OUTPUT_DIR}"
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
        "example_photon_output.cmake: run failed for backend=${BACKEND}.\n"
        "exit: ${run_result}\nstdout:\n${run_stdout}\nstderr:\n${run_stderr}"
    )
endif()

foreach(file metadata.csv outcomes.csv scatter_histogram.csv exits.csv traces.csv trace_summary.csv)
    if(NOT EXISTS "${OUTPUT_DIR}/${file}")
        message(FATAL_ERROR "example_photon_output.cmake: missing ${file}")
    endif()
endforeach()

file(STRINGS "${OUTPUT_DIR}/outcomes.csv" outcome_lines)
set(outcome_sum 0)
foreach(line IN LISTS outcome_lines)
    if(line MATCHES "^[a-z]+,([0-9]+)$")
        math(EXPR outcome_sum "${outcome_sum} + ${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT outcome_sum EQUAL 257)
    message(FATAL_ERROR "example_photon_output.cmake: outcome sum ${outcome_sum} != 257")
endif()

file(STRINGS "${OUTPUT_DIR}/trace_summary.csv" trace_lines)
list(LENGTH trace_lines trace_line_count)
if(NOT trace_line_count EQUAL 9)
    message(FATAL_ERROR "example_photon_output.cmake: expected 8 trace ids plus header, got ${trace_line_count}")
endif()

file(STRINGS "${OUTPUT_DIR}/traces.csv" vertex_lines)
list(LENGTH vertex_lines vertex_line_count)
if(vertex_line_count LESS 2)
    message(FATAL_ERROR "example_photon_output.cmake: traces.csv has no vertices")
endif()

file(STRINGS "${OUTPUT_DIR}/metadata.csv" metadata_lines)
set(has_commit FALSE)
set(has_pin FALSE)
set(has_trace FALSE)
foreach(line IN LISTS metadata_lines)
    if(line MATCHES "^graphflow_commit,")
        set(has_commit TRUE)
    endif()
    if(line MATCHES "^alpaka_pin,")
        set(has_pin TRUE)
    endif()
    if(line MATCHES "^trace_enabled,\"1\"$")
        set(has_trace TRUE)
    endif()
endforeach()
if(NOT has_commit OR NOT has_pin OR NOT has_trace)
    message(FATAL_ERROR "example_photon_output.cmake: metadata provenance/trace fields missing")
endif()

message(
    STATUS
    "example_photon_output: backend=${BACKEND} outcomes=${outcome_sum} traces=8 vertices>0"
)
