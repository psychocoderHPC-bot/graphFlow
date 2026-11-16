# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Source-level contract test for the public examples.
#
# The user-facing files must declare the graph through Flow handles and domain functors.
# They must not fall back to numeric node ids, route vectors, manual runner/pool setup, or
# the deprecated low-level example stages. EXAMPLE_KIND selects the expected public API:
# `particle` (default) or `record` for the explicit AoS-over-SoA example.

if(NOT DEFINED EXAMPLE_DIR)
    message(FATAL_ERROR "example_public_api.cmake: missing -DEXAMPLE_DIR=...")
endif()

if(NOT DEFINED EXAMPLE_KIND)
    set(EXAMPLE_KIND particle)
endif()

if(EXAMPLE_KIND STREQUAL "record")
    set(user_files main.cpp)
    set(forbidden
        "GraphRunner"
        "ChunkPool"
        "GraphBuilder"
        "addNode"
        "setSource"
        "setBatchChunks"
        "setStallTimeout"
        "NodeRole"
        "NodeId"
        "ParticleStorage"
        "DefaultStorage"
        "using SoAFlow"
    )
    set(required
        "graphFlow::Record<"
        "graphFlow::SoAFlow<"
        ".source("
        ".transform("
        ".partition("
        ".sink("
        ".connect("
    )
    set(graph_only_forbidden "")
elseif(EXAMPLE_KIND STREQUAL "particle")
    set(user_files
        ParticleFunctors.hpp
        ParticleTypes.hpp
        ParticleFeedback.hpp
        KernelAxes.hpp
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
    set(graph_only_forbidden
        "Config"
        "numChunks"
        "poolOutputReserveChunks"
        "NodeId"
    )
    set(required "")
else()
    message(FATAL_ERROR "example_public_api.cmake: unknown -DEXAMPLE_KIND='${EXAMPLE_KIND}'")
endif()

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
    if(file STREQUAL "ParticleFeedback.hpp")
        foreach(token IN LISTS graph_only_forbidden)
            string(FIND "${content}" "${token}" position)
            if(NOT position EQUAL -1)
                message(
                    FATAL_ERROR
                    "example_public_api.cmake: '${file}' contains forbidden graph-configuration token "
                    "'${token}'. The graph configuration must be derived by Flow."
                )
            endif()
        endforeach()
        # The public example payload is a Record run through the SoAFlow alias.
        string(FIND "${content}" "graphFlow::SoAFlow<" soaflow_position)
        if(soaflow_position EQUAL -1)
            message(
                FATAL_ERROR
                "example_public_api.cmake: '${file}' must run the public graph through 'graphFlow::SoAFlow<'."
            )
        endif()
    endif()
    foreach(token IN LISTS required)
        string(FIND "${content}" "${token}" position)
        if(position EQUAL -1)
            message(
                FATAL_ERROR
                "example_public_api.cmake: '${file}' is missing required public-API token '${token}'."
            )
        endif()
    endforeach()
endforeach()

message(
    STATUS
    "example_public_api (${EXAMPLE_KIND}): user code uses Flow handles and no low-level graph setup"
)
