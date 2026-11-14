/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonTypes.hpp"
#include "graphFlow/flow/Debug.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow::photon
{
    /** Component-wise finiteness of a three-vector. */
    ALPAKA_FN_ACC inline bool finite(Vec3 const& value)
    {
        return alpaka::math::isfinite(value.x) && alpaka::math::isfinite(value.y) && alpaka::math::isfinite(value.z);
    }

    /** Debug-only check of an in-flight photon state.
     *
     * The release build compiles the body to `true`; the device hot loop keeps no per-event state
     * branch. Host setup already validated the slab and the configuration.
     */
    ALPAKA_FN_ACC inline bool validatePhoton(Slab const& slab, Photon const& photon)
    {
        if constexpr(!graphFlow::debug::checksEnabled)
            return true;
        else
            return finite(photon.position) && finite(photon.direction) && photon.position.z >= 0.0
                   && photon.position.z <= slab.thickness;
    }

    /** Debug-only check that a proposal is finite and consistent with its event.
     *
     * Face events must land on their plane after the proposed distance. Collision events must
     * remain inside the slab. The release build compiles the body to `true`.
     */
    ALPAKA_FN_ACC inline bool validateProposal(Slab const& slab, Photon const& photon)
    {
        if constexpr(!graphFlow::debug::checksEnabled)
            return true;
        else
        {
            Proposal const& proposal = photon.proposal;
            if(!proposal.valid)
                return false;
            if(!alpaka::math::isfinite(proposal.distance) || proposal.distance < 0.0)
                return false;
            if(static_cast<std::uint32_t>(proposal.event) >= eventBins)
                return false;

            double const scale = slab.thickness > 1.0 ? slab.thickness : 1.0;
            double const tolerance = 1e-12 * scale;
            double const nextZ = photon.position.z + proposal.distance * photon.direction.z;
            switch(proposal.event)
            {
            case Event::transmitted:
                return photon.direction.z > 0.0 && nextZ >= slab.thickness - tolerance
                       && nextZ <= slab.thickness + tolerance;
            case Event::reflected:
                return photon.direction.z < 0.0 && nextZ >= -tolerance && nextZ <= tolerance;
            default:
                return nextZ >= -tolerance && nextZ <= slab.thickness + tolerance;
            }
        }
    }
} // namespace graphFlow::photon
