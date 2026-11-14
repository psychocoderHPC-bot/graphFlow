# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Source-level contract test for the public photon-scattering example.

if(NOT DEFINED EXAMPLE_DIR)
    message(FATAL_ERROR "example_photon_public_api.cmake: missing -DEXAMPLE_DIR=...")
endif()

set(user_files
    PhotonTypes.hpp
    PhotonFunctors.hpp
    PhotonScattering.hpp
    PhotonCli.hpp
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

foreach(file IN LISTS user_files)
    set(path "${EXAMPLE_DIR}/${file}")
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "example_photon_public_api.cmake: missing user file '${path}'")
    endif()
    file(READ "${path}" content)
    foreach(token IN LISTS forbidden)
        string(FIND "${content}" "${token}" position)
        if(NOT position EQUAL -1)
            message(
                FATAL_ERROR
                "example_photon_public_api.cmake: '${file}' contains forbidden low-level token '${token}'."
            )
        endif()
    endforeach()
    if(file STREQUAL "PhotonScattering.hpp")
        foreach(token IN LISTS graph_only_forbidden)
            string(FIND "${content}" "${token}" position)
            if(NOT position EQUAL -1)
                message(
                    FATAL_ERROR
                    "example_photon_public_api.cmake: '${file}' contains forbidden graph-configuration token "
                    "'${token}'."
                )
            endif()
        endforeach()
    endif()
endforeach()

message(STATUS "example_photon_public_api: user code uses Flow handles and named event ports")
