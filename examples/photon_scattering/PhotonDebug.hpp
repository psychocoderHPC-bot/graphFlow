/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonTypes.hpp"
#include "graphFlow/flow/Debug.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace graphFlow::photon
{
    /** Component-wise finiteness of a three-vector. */
    template<typename T>
    ALPAKA_FN_ACC inline bool finite(Vec3T<T> const& value)
    {
        return alpaka::math::isfinite(value.x) && alpaka::math::isfinite(value.y) && alpaka::math::isfinite(value.z);
    }

    /** Debug-only check of an in-flight photon state.
     *
     * The release build compiles the body to `true`; the device hot loop keeps no per-event state
     * branch. Host setup already validated the slab and the configuration.
     */
    template<typename T>
    ALPAKA_FN_ACC inline bool validatePhoton(SlabT<T> const& slab, PhotonT<T> const& photon)
    {
        if constexpr(!graphFlow::debug::checksEnabled)
            return true;
        else
            return finite(photon.position) && finite(photon.direction) && photon.position.z >= T{0}
                   && photon.position.z <= slab.thickness;
    }

    /** Debug-only check that a proposal is finite and consistent with its event.
     *
     * Face events must land on their plane after the proposed distance. Collision events must
     * remain inside the slab. The release build compiles the body to `true`.
     */
    template<typename T>
    ALPAKA_FN_ACC inline bool validateProposal(SlabT<T> const& slab, PhotonT<T> const& photon)
    {
        if constexpr(!graphFlow::debug::checksEnabled)
            return true;
        else
        {
            ProposalT<T> const& proposal = photon.proposal;
            if(!proposal.valid)
                return false;
            if(!alpaka::math::isfinite(proposal.distance) || proposal.distance < T{0})
                return false;
            if(static_cast<std::uint32_t>(proposal.event) >= eventBins)
                return false;

            T const scale = slab.thickness > T{1} ? slab.thickness : T{1};
            T const tolerance
                = std::is_same_v<T, double> ? T{1e-12} * scale : T{64} * std::numeric_limits<T>::epsilon() * scale;
            T const nextZ = photon.position.z + proposal.distance * photon.direction.z;
            switch(proposal.event)
            {
            case Event::transmitted:
                return photon.direction.z > T{0} && nextZ >= slab.thickness - tolerance
                       && nextZ <= slab.thickness + tolerance;
            case Event::reflected:
                return photon.direction.z < T{0} && nextZ >= -tolerance && nextZ <= tolerance;
            default:
                return nextZ >= -tolerance && nextZ <= slab.thickness + tolerance;
            }
        }
    }
} // namespace graphFlow::photon
