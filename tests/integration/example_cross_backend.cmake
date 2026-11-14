# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Cross-backend histogram equality for the graphFlow_particle_feedback example.
#
# The driver runs the real executable once per requested backend with identical
# seed/N/C/P/Q, parses the `# weight,count` histogram, sorts the rows and checks
# that every backend produces exactly the same multiset of rows and that every
# count sum equals N. A missing binary, a missing device, or a non-zero exit is
# fatal, so a compiled backend that cannot run fails the case instead of being
# skipped.

foreach(required BINARY BACKENDS PARTICLES CHUNK_SIZE LIVE_PARTICLES BATCH_CHUNKS SEED MAX_VISITS)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "example_cross_backend.cmake: missing -D${required}=...")
    endif()
endforeach()

if(NOT DEFINED TIMEOUT)
    set(TIMEOUT 900)
endif()

if(NOT EXISTS "${BINARY}")
    message(
        FATAL_ERROR
        "example_cross_backend.cmake: example binary '${BINARY}' does not exist. "
        "Build the graphFlow_particle_feedback target (GRAPHFLOW_BUILD_EXAMPLES=ON) before running CTest."
    )
endif()

string(REPLACE "," ";" backend_list "${BACKENDS}")
list(LENGTH backend_list backend_count)
if(backend_count LESS 1)
    message(FATAL_ERROR "example_cross_backend.cmake: BACKENDS='${BACKENDS}' is empty")
endif()

set(reference_rows "")
set(reference_backend "")
set(reference_row_count -1)
set(reference_count_sum -1)

foreach(backend IN LISTS backend_list)
    set(command
        "${BINARY}"
        --backend
        "${backend}"
        --particles
        "${PARTICLES}"
        --chunk-size
        "${CHUNK_SIZE}"
        --live-particles
        "${LIVE_PARTICLES}"
        --batch-chunks
        "${BATCH_CHUNKS}"
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
        TIMEOUT ${TIMEOUT}
    )

    if(NOT "${run_result}" STREQUAL "0")
        message(
            FATAL_ERROR
            "example_cross_backend.cmake: '${BINARY}' failed for backend=${backend} N=${PARTICLES} "
            "C=${CHUNK_SIZE} P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS} seed=${SEED}.\n"
            "exit: ${run_result}\n"
            "stdout:\n${run_stdout}\n"
            "stderr:\n${run_stderr}"
        )
    endif()

    # Reject a silent fallback to another backend and confirm the requested
    # configuration reached the device policy unchanged.
    set(expected_run_line "# graphFlow_particle_feedback backend=${backend} device=0 ")
    string(FIND "${run_stdout}" "${expected_run_line}" run_line_position)
    if(run_line_position EQUAL -1)
        message(
            FATAL_ERROR
            "example_cross_backend.cmake: no example header for backend '${backend}'.\n"
            "stdout:\n${run_stdout}\n"
            "stderr:\n${run_stderr}"
        )
    endif()
    foreach(token
        " particles=${PARTICLES} "
        " chunk-size=${CHUNK_SIZE} "
        " live-particles=${LIVE_PARTICLES} "
        " batch-chunks=${BATCH_CHUNKS} "
        " seed=${SEED} "
    )
        string(FIND "${run_stdout}" "${token}" token_position)
        if(token_position EQUAL -1)
            message(
                FATAL_ERROR
                "example_cross_backend.cmake: header for backend=${backend} misses '${token}'.\n"
                "stdout:\n${run_stdout}\n"
                "stderr:\n${run_stderr}"
            )
        endif()
    endforeach()

    string(REPLACE "\r" "" run_stdout "${run_stdout}")
    string(REPLACE "\n" ";" output_lines "${run_stdout}")

    set(rows "")
    set(in_histogram FALSE)
    set(saw_histogram FALSE)
    set(count_sum 0)
    set(non_terminating 0)

    foreach(line IN LISTS output_lines)
        if(line MATCHES "^# non-terminating ([0-9]+)$")
            math(EXPR non_terminating "${CMAKE_MATCH_1}")
            continue()
        endif()
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
                "example_cross_backend.cmake: malformed histogram row '${line}' for backend=${backend} "
                "N=${PARTICLES} C=${CHUNK_SIZE} P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS}."
            )
        endif()
        list(APPEND rows "${line}")
        math(EXPR count_sum "${count_sum} + ${CMAKE_MATCH_2}")
    endforeach()

    if(NOT saw_histogram)
        message(
            FATAL_ERROR
            "example_cross_backend.cmake: no '# weight,count' header for backend=${backend}.\n"
            "stdout:\n${run_stdout}\n"
            "stderr:\n${run_stderr}"
        )
    endif()
    if(NOT count_sum EQUAL PARTICLES)
        message(
            FATAL_ERROR
            "example_cross_backend.cmake: histogram counts sum to ${count_sum}, expected ${PARTICLES} "
            "for backend=${backend} C=${CHUNK_SIZE} P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS}.\n"
            "non-terminating: ${non_terminating}\n"
            "stdout:\n${run_stdout}\n"
            "stderr:\n${run_stderr}"
        )
    endif()

    list(SORT rows)
    list(LENGTH rows row_count)

    if(reference_backend STREQUAL "")
        set(reference_rows "${rows}")
        set(reference_backend "${backend}")
        set(reference_row_count "${row_count}")
        set(reference_count_sum "${count_sum}")
        message(
            STATUS
            "example_cross_backend: backend=${backend} N=${PARTICLES} C=${CHUNK_SIZE} "
            "P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS} rows=${row_count} count-sum=${count_sum}"
        )
        continue()
    endif()

    # Find the first sorted row that differs, then report a bounded diagnostic
    # instead of dumping a potentially large histogram.
    set(mismatch_index -1)
    set(common_rows "${row_count}")
    if(reference_row_count LESS common_rows)
        set(common_rows "${reference_row_count}")
    endif()
    set(index 0)
    while(index LESS common_rows)
        list(GET reference_rows ${index} reference_row)
        list(GET rows ${index} row)
        if(NOT "${reference_row}" STREQUAL "${row}")
            set(mismatch_index "${index}")
            break()
        endif()
        math(EXPR index "${index} + 1")
    endwhile()
    if(mismatch_index EQUAL -1 AND NOT reference_row_count EQUAL row_count)
        set(mismatch_index "${common_rows}")
    endif()

    if(NOT mismatch_index EQUAL -1)
        set(reference_value "<missing>")
        set(value "<missing>")
        if(mismatch_index LESS reference_row_count)
            list(GET reference_rows ${mismatch_index} reference_value)
        endif()
        if(mismatch_index LESS row_count)
            list(GET rows ${mismatch_index} value)
        endif()
        message(
            FATAL_ERROR
            "example_cross_backend.cmake: histogram mismatch for N=${PARTICLES} C=${CHUNK_SIZE} "
            "P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS} seed=${SEED}.\n"
            "${reference_backend}: rows=${reference_row_count} count-sum=${reference_count_sum}\n"
            "${backend}: rows=${row_count} count-sum=${count_sum}\n"
            "first sorted-row difference at index ${mismatch_index}: "
            "${reference_backend}='${reference_value}' ${backend}='${value}'"
        )
    endif()

    message(
        STATUS
        "example_cross_backend: backend=${backend} N=${PARTICLES} C=${CHUNK_SIZE} "
        "P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS} rows=${row_count} count-sum=${count_sum} exact-match"
    )
endforeach()

string(REPLACE ";" "," backend_label "${backend_list}")
message(
    STATUS
    "example_cross_backend: backends=${backend_label} N=${PARTICLES} C=${CHUNK_SIZE} "
    "P=${LIVE_PARTICLES} Q=${BATCH_CHUNKS} rows=${reference_row_count} "
    "count-sum=${reference_count_sum} exact-row-equality=yes"
)
