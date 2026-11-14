/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonDebug.hpp"
#include "PhotonTypes.hpp"
#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>

namespace graphFlow::photon
{
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

    /** Resolve one already-sampled flight into a proposal.
     *
     * `sigmaTotalValue` is the host-validated `sigmaTotal(slab)`, passed in so the hot loop adds it
     * once per event. `sampledDistance` is `-log(U) / sigmaTotalValue` when material is present and
     * is ignored for a ballistic vacuum ray. `sampledKind` is the independent collision-kind
     * uniform. The function is pure so geometry, tie and cap decisions are testable without a graph
     * or an RNG.
     *
     * Release builds trust the host-validated slab and photon state. `graphFlow::debug::checksEnabled`
     * adds the finite/range checks in `PhotonDebug.hpp`; the physics errors stay unconditional: a
     * vacuum ray with no reachable face is an invalid proposal.
     */
    ALPAKA_FN_ACC inline Proposal resolveFlight(
        Slab const& slab,
        double sigmaTotalValue,
        Photon const& photon,
        double sampledDistance,
        double sampledKind,
        std::uint32_t maxScatters)
    {
        Proposal proposal;
        proposal.event = Event::absorbed;
        proposal.distance = 0.0;
        proposal.valid = false;

        if(!validatePhoton(slab, photon))
            return proposal;

        FaceHit const face = faceHit(slab, photon.position, photon.direction);

        if(sigmaTotalValue == 0.0)
        {
            // Ballistic vacuum ray. A parallel ray never reaches a face in this model.
            if(!face.reachable)
                return proposal;
            proposal.event = face.event;
            proposal.distance = face.distance;
            proposal.valid = true;
            return proposal;
        }

        if(face.reachable && faceFirst(face.distance, sampledDistance))
        {
            proposal.event = face.event;
            proposal.distance = face.distance;
            proposal.valid = true;
            return proposal;
        }

        bool const scatters = sampledKind < slab.sigmaS / sigmaTotalValue;
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
        if(!validateProposal(slab, photon))
            return UpdateResult::invalid;
        double const distance = photon.proposal.distance;

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

        if(!validatePhoton(slab, photon))
            return UpdateResult::invalid;
        return UpdateResult::success;
    }

    /** Isotropic direction from the scatter engine.
     *
     * The cosine draw uses `[0, 1)`; `2U - 1` covers `[-1, 1)`. The azimuth uses a second `[0, 1)`
     * draw. graphFlow owns the event and purpose, so these are consecutive draws of the scatter
     * stage engine.
     */
    ALPAKA_FN_ACC inline Vec3 isotropicDirection(rng::Engine& engine)
    {
        double const mu = 2.0 * alpaka::rand::distribution::UniformReal<double>{}(engine) -1.0;
        double const phi
            = 2.0 * 3.141592653589793238462643383279502884 * alpaka::rand::distribution::UniformReal<double>{}(engine);
        double const transverse = alpaka::math::sqrt(alpaka::math::max(0.0, 1.0 - mu * mu));
        return Vec3{transverse * alpaka::math::cos(phi), transverse * alpaka::math::sin(phi), mu};
    }

    /** No-op trace writer used by production runs without diagnostics. */
    struct NoTrace
    {
        ALPAKA_FN_ACC void record(
            [[maybe_unused]] std::uint64_t id,
            [[maybe_unused]] TraceKind kind,
            [[maybe_unused]] std::uint32_t scatterCount,
            [[maybe_unused]] Vec3 const& position,
            [[maybe_unused]] Vec3 const& direction) const
        {
        }
    };

    /** Bounded deterministic trace writer.
     *
     * Selected ids are exactly `id < limit`. Each id owns `maxVertices` contiguous slots; a write
     * beyond the bound sets the per-id truncation flag and never touches the simulation state.
     */
    template<typename TVertices, typename TCounts, typename TFlags>
    struct TraceWriter
    {
        mutable TVertices vertices;
        mutable TCounts counts;
        mutable TFlags truncated;
        std::uint32_t maxVertices = 0u;
        std::uint32_t limit = 0u;

        ALPAKA_FN_ACC void record(
            std::uint64_t id,
            TraceKind kind,
            std::uint32_t scatterCount,
            Vec3 const& position,
            Vec3 const& direction) const
        {
            if(id >= limit)
                return;
            std::uint32_t const index = counts[id];
            if(index >= maxVertices)
            {
                truncated[id] = 1u;
                return;
            }
            vertices[id * maxVertices + index] = TraceVertex{kind, scatterCount, position, direction};
            counts[id] = index + 1u;
        }
    };

    /** Map a terminal event onto the diagnostic vertex kind. */
    ALPAKA_FN_ACC inline TraceKind traceKindOf(Event event)
    {
        return static_cast<TraceKind>(static_cast<std::uint32_t>(event) + 1u);
    }

    /** Source: one independent history per logical id at the entrance plane, normal incidence. */
    template<typename TTrace = NoTrace>
    struct GeneratePhotonT
    {
        mutable TTrace trace;

        ALPAKA_FN_ACC Photon operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            Photon photon{id, Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 0.0, 1.0}, 0u, Proposal{}};
            trace.record(id, TraceKind::source, 0u, photon.position, photon.direction);
            return photon;
        }
    };

    using GeneratePhoton = GeneratePhotonT<>;

    /** Transform: draw the flight and collision-kind uniforms and store a proposal.
     *
     * The flight sample needs a positive value for `log`. `UniformReal<double, interval::CO>`
     * returns `[0, 1)`; `1 - U` maps it to `(0, 1]`, so the material distance is non-negative. The
     * collision-kind draw is a second `[0, 1)` sample from the same propose-stage engine. The host
     * validated the slab once, so the hot loop keeps only the vacuum/material branch and the
     * optional debug proposal check.
     */
    struct ProposeFlight
    {
        Slab slab{};
        std::uint32_t maxScatters = 0u;

        ALPAKA_FN_ACC void operator()(auto const&, rng::Engine& engine, Status& status, Photon& photon) const
        {
            double const flightUniform = 1.0 - alpaka::rand::distribution::UniformReal<double>{}(engine);
            double const kindUniform = alpaka::rand::distribution::UniformReal<double>{}(engine);
            double const total = sigmaTotal(slab);
            double sampledDistance = 0.0;
            if(total > 0.0)
                sampledDistance = -alpaka::math::log(flightUniform) / total;
            photon.proposal = resolveFlight(slab, total, photon, sampledDistance, kindUniform, maxScatters);
            if(!photon.proposal.valid || !validateProposal(slab, photon))
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
    template<typename TTrace = NoTrace>
    struct ScatterT
    {
        Slab slab{};
        ScatterMode mode = ScatterMode::forward;
        mutable TTrace trace;

        ALPAKA_FN_ACC void operator()(auto const&, rng::Engine& engine, Status& status, Photon& photon) const
        {
            if(!photon.proposal.valid || photon.proposal.event != Event::scatter)
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(mode == ScatterMode::isotropic)
                photon.direction = isotropicDirection(engine);
            if(!validatePhoton(slab, photon))
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(photon.scatterCount == std::numeric_limits<std::uint32_t>::max())
            {
                status.set(StatusCode::overflow);
                return;
            }
            trace.record(photon.id, TraceKind::scatter, photon.scatterCount + 1u, photon.position, photon.direction);
            ++photon.scatterCount;
        }
    };

    using Scatter = ScatterT<>;

    /** Shared terminal sink: record one final state, trace vertex and per-id delivery count. */
    template<typename TRecords, typename TVisits, typename TTrace = NoTrace>
    struct RecordTerminal
    {
        mutable TRecords records;
        mutable TVisits visits;
        mutable TTrace trace;

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

            records[photon.id] = TerminalRecord{event, photon.scatterCount, photon.position, photon.direction};
            trace.record(photon.id, traceKindOf(event), photon.scatterCount, photon.position, photon.direction);
            std::uint32_t const previous
                = alpaka::onAcc::atomicAdd(acc, &visits[photon.id], std::uint32_t{1u}, alpaka::onAcc::scope::device);
            if(previous != 0u)
                status.set(StatusCode::invalid);
        }
    };
} // namespace graphFlow::photon
