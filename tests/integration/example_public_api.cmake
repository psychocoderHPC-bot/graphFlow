# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Source-level contract test for the public particle-feedback example.
#
# The user-facing files must declare the graph through Flow handles and domain functors.
# They must not fall back to numeric node ids, route vectors, manual runner/pool setup, or
# the deprecated low-level example stages.

if(NOT DEFINED EXAMPLE_DIR)
    message(FATAL_ERROR "example_public_api.cmake: missing -DEXAMPLE_DIR=...")
endif()

set(user_files
    ParticleFunctors.hpp
    ParticleTypes.hpp
    ParticleFeedback.hpp
    Cli.hpp
    main.cpp
)

set(forbidden
    "hooksOf"
    "derivePoolSize"
    "defaultLiveParticles"
    "GraphRunner"
    "ChunkPool"
    "GraphBuilder"
    "addNode"
    "setSource"
    "setBatchChunks"
    "setStallTimeout"
    "NodeRole"
    "ParticleStages.hpp"
)

foreach(file IN LISTS user_files)
    set(path "${EXAMPLE_DIR}/${file}")
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "example_public_api.cmake: missing user file '${path}'")
    endif()
    file(READ "${path}" content)
    foreach(token IN LISTS forbidden)
        string(FIND "${content}" "${token}" position)
        if(NOT position EQUAL -1)
            message(
                FATAL_ERROR
                "example_public_api.cmake: '${file}' contains forbidden low-level token '${token}'. "
                "The example must use Flow handles and named outputs."
            )
        endif()
    endforeach()
endforeach()

message(STATUS "example_public_api: user code uses Flow handles and no low-level graph setup")
