/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonTypes.hpp"

#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>

namespace graphFlow::photon
{
    /** Open-interval uniform from one 32 bit Philox word.
     *
     * The midpoint mapping `(word + 0.5) * 2^-32` is strictly inside `(0, 1)`, so `log(U)` is
     * finite and the exponential free-flight sample is always positive.
     */
    ALPAKA_FN_ACC inline double openUniform(std::uint32_t word)
    {
        return (static_cast<double>(word) + 0.5) * (1.0 / 4294967296.0);
    }

    ALPAKA_FN_ACC inline bool finite(Vec3 const& value)
    {
        return alpaka::math::isfinite(value.x) && alpaka::math::isfinite(value.y)
               && alpaka::math::isfinite(value.z);
    }

    /** Face reached by the current ray, if any. */
    struct FaceHit
    {
        bool reachable = false;
        double distance = 0.0;
        Event event = Event::absorbed;
    };

    /** Distance to the next slab face for a photon inside `[0, L]`.
     *
     * A positive `uz` sees the exit plane `z = L`; a negative `uz` sees the entrance plane
     * `z = 0`. A ray parallel to the planes (`uz == 0`) has no reachable face in this infinite-slab
     * model; in vacuum that is an explicit error, while in material it collides eventually.
     */
    ALPAKA_FN_ACC inline FaceHit faceHit(Slab const& slab, Vec3 const& position, Vec3 const& direction)
    {
        FaceHit hit;
        if(direction.z > 0.0)
        {
            hit.reachable = true;
            hit.distance = (slab.thickness - position.z) / direction.z;
            hit.event = Event::transmitted;
        }
        else if(direction.z < 0.0)
        {
            hit.reachable = true;
            hit.distance = -position.z / direction.z;
            hit.event = Event::reflected;
        }
        return hit;
    }

    /** Exact face-priority rule: equality belongs to the face, not the collision. */
    ALPAKA_FN_ACC inline bool faceFirst(double faceDistance, double sampledDistance)
    {
        return faceDistance <= sampledDistance;
    }

    /** Resolve one already-sampled flight into a checked proposal.
     *
     * `sampledDistance` is `-log(U) / sigmaT` when material is present and is ignored for a
     * ballistic vacuum ray. `sampledKind` is the independent collision-kind uniform. The function
     * is pure so geometry, tie and cap decisions are testable without a graph or an RNG.
     */
    ALPAKA_FN_ACC inline Proposal resolveFlight(
        Slab const& slab,
        Photon const& photon,
        double sampledDistance,
        double sampledKind,
        std::uint32_t maxScatters)
    {
        Proposal proposal;
        proposal.event = Event::absorbed;
        proposal.distance = 0.0;
        proposal.valid = false;

        if(!finite(photon.position) || !finite(photon.direction))
            return proposal;
        if(!alpaka::math::isfinite(slab.thickness) || slab.thickness <= 0.0)
            return proposal;
        if(photon.position.z < 0.0 || photon.position.z > slab.thickness)
            return proposal;

        double const total = sigmaTotal(slab);
        if(!alpaka::math::isfinite(total) || total < 0.0)
            return proposal;

        FaceHit const face = faceHit(slab, photon.position, photon.direction);
        if(face.reachable && (!alpaka::math::isfinite(face.distance) || face.distance < 0.0))
            return proposal;

        if(total == 0.0)
        {
            // Ballistic vacuum ray. A parallel ray never reaches a face in this model.
            if(!face.reachable)
                return proposal;
            proposal.event = face.event;
            proposal.distance = face.distance;
            proposal.valid = true;
            return proposal;
        }

        if(!alpaka::math::isfinite(sampledDistance) || sampledDistance <= 0.0)
            return proposal;
        if(!alpaka::math::isfinite(sampledKind) || sampledKind < 0.0 || sampledKind >= 1.0)
            return proposal;

        if(face.reachable && faceFirst(face.distance, sampledDistance))
        {
            proposal.event = face.event;
            proposal.distance = face.distance;
            proposal.valid = true;
            return proposal;
        }

        bool const scatters = sampledKind < slab.sigmaS / total;
        proposal.distance = sampledDistance;
        if(!scatters)
            proposal.event = Event::absorbed;
        else if(photon.scatterCount >= maxScatters)
            proposal.event = Event::capped;
        else
            proposal.event = Event::scatter;
        proposal.valid = true;
        return proposal;
    }

    /** Move the photon to the proposed event.
     *
     * Face events clamp `z` to the exact plane after advancing `x` and `y` along the ray. Collision
     * events advance along the full sampled path. The proposal and the selected bin must agree.
     */
    ALPAKA_FN_ACC inline UpdateResult applyEvent(Slab const& slab, Event event, Photon& photon)
    {
        if(!photon.proposal.valid || event != photon.proposal.event)
            return UpdateResult::invalid;
        double const distance = photon.proposal.distance;
        if(!alpaka::math::isfinite(distance) || distance < 0.0)
            return UpdateResult::invalid;
        if(!finite(photon.direction))
            return UpdateResult::invalid;

        switch(event)
        {
        case Event::scatter:
        case Event::absorbed:
        case Event::capped:
            photon.position.x += distance * photon.direction.x;
            photon.position.y += distance * photon.direction.y;
            photon.position.z += distance * photon.direction.z;
            break;
        case Event::transmitted:
            if(photon.direction.z <= 0.0)
                return UpdateResult::invalid;
            photon.position.x += distance * photon.direction.x;
            photon.position.y += distance * photon.direction.y;
            photon.position.z = slab.thickness;
            break;
        case Event::reflected:
            if(photon.direction.z >= 0.0)
                return UpdateResult::invalid;
            photon.position.x += distance * photon.direction.x;
            photon.position.y += distance * photon.direction.y;
            photon.position.z = 0.0;
            break;
        }

        if(!finite(photon.position))
            return UpdateResult::invalid;
        return UpdateResult::success;
    }

    /** Isotropic direction from the cosine and azimuth purposes at the current scatter count. */
    ALPAKA_FN_ACC inline Vec3 isotropicDirection(
        std::uint64_t seed,
        std::uint64_t id,
        std::uint32_t scatterCount)
    {
        std::uint32_t const cosineWord = rng::draw(
            seed,
            id,
            scatterCount,
            rng::purposeCosine,
            rng::photonPurposesPerVisit);
        std::uint32_t const azimuthWord = rng::draw(
            seed,
            id,
            scatterCount,
            rng::purposeAzimuth,
            rng::photonPurposesPerVisit);
        double const mu = 2.0 * openUniform(cosineWord) - 1.0;
        double const phi = 2.0 * 3.141592653589793238462643383279502884 * openUniform(azimuthWord);
        double const transverse = alpaka::math::sqrt(alpaka::math::max(0.0, 1.0 - mu * mu));
        return Vec3{
            transverse * alpaka::math::cos(phi),
            transverse * alpaka::math::sin(phi),
            mu};
    }

    /** Source: one independent history per logical id at the entrance plane, normal incidence. */
    struct GeneratePhoton
    {
        ALPAKA_FN_ACC Photon operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Photon{id, Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 0.0, 1.0}, 0u, Proposal{}};
        }
    };

    /** Transform: draw the flight and collision-kind uniforms and store a checked proposal. */
    struct ProposeFlight
    {
        Slab slab{};
        std::uint64_t seed = 0u;
        std::uint32_t maxScatters = 0u;

        ALPAKA_FN_ACC void operator()(auto const&, Status& status, Photon& photon) const
        {
            std::uint32_t const flightWord = rng::draw(
                seed,
                photon.id,
                photon.scatterCount,
                rng::purposeFlight,
                rng::photonPurposesPerVisit);
            std::uint32_t const kindWord = rng::draw(
                seed,
                photon.id,
                photon.scatterCount,
                rng::purposeCollisionKind,
                rng::photonPurposesPerVisit);
            double const uniform = openUniform(flightWord);
            double const total = sigmaTotal(slab);
            double sampledDistance = 0.0;
            if(alpaka::math::isfinite(total) && total > 0.0)
                sampledDistance = -alpaka::math::log(uniform) / total;
            photon.proposal = resolveFlight(
                slab,
                photon,
                sampledDistance,
                openUniform(kindWord),
                maxScatters);
            if(!photon.proposal.valid)
                status.set(StatusCode::invalid);
        }
    };

    /** Classifier: read the proposal event, rejecting invalid proposals before the update. */
    struct ClassifyEvent
    {
        static constexpr std::uint32_t maxBins = eventBins;

        ALPAKA_FN_ACC Event operator()(auto const&, Status& status, Photon const& photon) const
        {
            if(!photon.proposal.valid)
            {
                status.set(StatusCode::invalid);
                return Event::absorbed;
            }
            std::uint32_t const bin = static_cast<std::uint32_t>(photon.proposal.event);
            if(bin >= maxBins)
            {
                status.set(StatusCode::invalid);
                return Event::absorbed;
            }
            return photon.proposal.event;
        }
    };

    /** Updater: advance to the selected event and report checked failures. */
    struct AdvanceToEvent
    {
        Slab slab{};

        ALPAKA_FN_ACC UpdateResult operator()(auto const&, Status& status, Event event, Photon& photon) const
        {
            UpdateResult const result = applyEvent(slab, event, photon);
            if(result == UpdateResult::invalid)
                status.set(StatusCode::invalid);
            else if(result == UpdateResult::overflow)
                status.set(StatusCode::overflow);
            return result;
        }
    };

    /** Transform: change direction and increment the completed-scatter counter exactly once. */
    struct Scatter
    {
        ScatterMode mode = ScatterMode::forward;
        std::uint64_t seed = 0u;

        ALPAKA_FN_ACC void operator()(auto const&, Status& status, Photon& photon) const
        {
            if(!photon.proposal.valid || photon.proposal.event != Event::scatter)
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(mode == ScatterMode::isotropic)
                photon.direction = isotropicDirection(seed, photon.id, photon.scatterCount);
            if(!finite(photon.direction))
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(photon.scatterCount == std::numeric_limits<std::uint32_t>::max())
            {
                status.set(StatusCode::overflow);
                return;
            }
            ++photon.scatterCount;
        }
    };

    /** Shared terminal sink: record one final state and count per-id deliveries. */
    template<typename TRecords, typename TVisits>
    struct RecordTerminal
    {
        mutable TRecords records;
        mutable TVisits visits;

        ALPAKA_FN_ACC void operator()(auto const& acc, Status& status, Photon const& photon) const
        {
            if(!photon.proposal.valid)
            {
                status.set(StatusCode::invalid);
                return;
            }
            Event const event = photon.proposal.event;
            std::uint32_t const bin = static_cast<std::uint32_t>(event);
            if(bin == static_cast<std::uint32_t>(Event::scatter) || bin >= eventBins)
            {
                status.set(StatusCode::invalid);
                return;
            }

            records[photon.id] = TerminalRecord{
                event,
                photon.scatterCount,
                photon.position,
                photon.direction};
            std::uint32_t const previous = alpaka::onAcc::atomicAdd(
                acc,
                &visits[photon.id],
                std::uint32_t{1u},
                alpaka::onAcc::scope::device);
            if(previous != 0u)
                status.set(StatusCode::invalid);
        }
    };
} // namespace graphFlow::photon
