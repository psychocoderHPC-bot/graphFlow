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
    template<typename T>
    struct FaceHitT
    {
        bool reachable = false;
        T distance = T{0};
        Event event = Event::absorbed;
    };

    /** Distance to the next slab face for a photon inside `[0, L]`.
     *
     * A positive `uz` sees the exit plane `z = L`; a negative `uz` sees the entrance plane
     * `z = 0`. A ray parallel to the planes (`uz == 0`) has no reachable face in this infinite-slab
     * model; in vacuum that is an explicit error, while in material it collides eventually.
     */
    template<typename T>
    ALPAKA_FN_ACC inline FaceHitT<T> faceHit(SlabT<T> const& slab, Vec3T<T> const& position, Vec3T<T> const& direction)
    {
        FaceHitT<T> hit;
        if(direction.z > T{0})
        {
            hit.reachable = true;
            hit.distance = (slab.thickness - position.z) / direction.z;
            hit.event = Event::transmitted;
        }
        else if(direction.z < T{0})
        {
            hit.reachable = true;
            hit.distance = -position.z / direction.z;
            hit.event = Event::reflected;
        }
        return hit;
    }

    /** Exact face-priority rule: equality belongs to the face, not the collision. */
    template<typename T>
    ALPAKA_FN_ACC inline bool faceFirst(T faceDistance, T sampledDistance)
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
    template<typename T>
    ALPAKA_FN_ACC inline ProposalT<T> resolveFlight(
        SlabT<T> const& slab,
        T sigmaTotalValue,
        PhotonT<T> const& photon,
        T sampledDistance,
        T sampledKind,
        std::uint32_t maxScatters)
    {
        ProposalT<T> proposal;
        proposal.event = Event::absorbed;
        proposal.distance = T{0};
        proposal.valid = false;

        if(!validatePhoton(slab, photon))
            return proposal;

        Vec3T<T> const& position = photon[positionTag];
        Vec3T<T> const& direction = photon[directionTag];
        FaceHitT<T> const face = faceHit(slab, position, direction);

        if(sigmaTotalValue == T{0})
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
        else if(photon[scatterCountTag] >= maxScatters)
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
    template<typename T>
    ALPAKA_FN_ACC inline UpdateResult applyEvent(SlabT<T> const& slab, Event event, PhotonT<T>& photon)
    {
        ProposalT<T> const& proposal = photon[proposalTag];
        if(!proposal.valid || event != proposal.event)
            return UpdateResult::invalid;
        if(!validateProposal(slab, photon))
            return UpdateResult::invalid;
        T const distance = proposal.distance;
        Vec3T<T>& position = photon[positionTag];
        Vec3T<T> const& direction = photon[directionTag];

        switch(event)
        {
        case Event::scatter:
        case Event::absorbed:
        case Event::capped:
            position.x += distance * direction.x;
            position.y += distance * direction.y;
            position.z += distance * direction.z;
            break;
        case Event::transmitted:
            if(direction.z <= T{0})
                return UpdateResult::invalid;
            position.x += distance * direction.x;
            position.y += distance * direction.y;
            position.z = slab.thickness;
            break;
        case Event::reflected:
            if(direction.z >= 0.0)
                return UpdateResult::invalid;
            position.x += distance * direction.x;
            position.y += distance * direction.y;
            position.z = T{0};
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
    template<typename T>
    ALPAKA_FN_ACC inline Vec3T<T> isotropicDirection(rng::Engine& engine)
    {
        T const pi = static_cast<T>(alpaka::math::constants::pi);
        T const mu = T{2} * alpaka::rand::distribution::UniformReal<T>{}(engine) -T{1};
        T const phi = T{2} * pi * alpaka::rand::distribution::UniformReal<T>{}(engine);
        T const transverse = alpaka::math::sqrt(alpaka::math::max(T{0}, T{1} - mu * mu));
        return Vec3T<T>{transverse * alpaka::math::cos(phi), transverse * alpaka::math::sin(phi), mu};
    }

    /** No-op trace writer used by production runs without diagnostics. */
    struct NoTrace
    {
        ALPAKA_FN_ACC void record(
            [[maybe_unused]] std::uint64_t id,
            [[maybe_unused]] TraceKind kind,
            [[maybe_unused]] std::uint32_t scatterCount,
            [[maybe_unused]] auto const& position,
            [[maybe_unused]] auto const& direction) const
        {
        }
    };

    /** Bounded deterministic trace writer.
     *
     * Selected ids are exactly `id < limit`. Each id owns `maxVertices` contiguous slots; a write
     * beyond the bound sets the per-id truncation flag and never touches the simulation state.
     */
    template<typename TVertices, typename TCounts, typename TFlags, typename T = double>
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
            Vec3T<T> const& position,
            Vec3T<T> const& direction) const
        {
            if(id >= limit)
                return;
            std::uint32_t const index = counts[id];
            if(index >= maxVertices)
            {
                truncated[id] = 1u;
                return;
            }
            vertices[id * maxVertices + index] = TraceVertexT<T>{kind, scatterCount, position, direction};
            counts[id] = index + 1u;
        }
    };

    /** Map a terminal event onto the diagnostic vertex kind. */
    ALPAKA_FN_ACC inline TraceKind traceKindOf(Event event)
    {
        return static_cast<TraceKind>(static_cast<std::uint32_t>(event) + 1u);
    }

    /** Source: one independent history per logical id at the entrance plane, normal incidence. */
    template<typename TTrace = NoTrace, typename T = double>
    struct GeneratePhotonT
    {
        mutable TTrace trace;

        ALPAKA_FN_ACC PhotonT<T> operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            PhotonT<T> photon{};
            photon[idTag] = id;
            photon[positionTag] = Vec3T<T>{T{0}, T{0}, T{0}};
            photon[directionTag] = Vec3T<T>{T{0}, T{0}, T{1}};
            photon[scatterCountTag] = 0u;
            photon[proposalTag] = ProposalT<T>{};
            trace.record(id, TraceKind::source, 0u, photon[positionTag], photon[directionTag]);
            return photon;
        }
    };

    using GeneratePhoton = GeneratePhotonT<>;

    /** Transform: draw the flight and collision-kind uniforms and store a proposal.
     *
     * The flight sample needs a positive value for `log`. `UniformReal<T, interval::CO>`
     * returns `[0, 1)`; `1 - U` maps it to `(0, 1]`, so the material distance is non-negative. The
     * collision-kind draw is a second `[0, 1)` sample from the same propose-stage engine. The host
     * validated the slab once, so the hot loop keeps only the vacuum/material branch and the
     * optional debug proposal check.
     */
    template<typename T = double>
    struct ProposeFlightT
    {
        SlabT<T> slab{};
        std::uint32_t maxScatters = 0u;

        ALPAKA_FN_ACC void operator()(auto const&, rng::Engine& engine, Status& status, PhotonT<T>& photon) const
        {
            T const flightUniform = T{1} - alpaka::rand::distribution::UniformReal<T>{}(engine);
            T const kindUniform = alpaka::rand::distribution::UniformReal<T>{}(engine);
            T const total = sigmaTotal(slab);
            T sampledDistance = T{0};
            if(total > T{0})
                sampledDistance = -alpaka::math::log(flightUniform) / total;
            photon[proposalTag] = resolveFlight(slab, total, photon, sampledDistance, kindUniform, maxScatters);
            if(!photon[proposalTag].valid || !validateProposal(slab, photon))
                status.set(StatusCode::invalid);
        }
    };

    using ProposeFlight = ProposeFlightT<>;

    /** Classifier: read the proposal event, rejecting invalid proposals before the update. */
    template<typename T = double>
    struct ClassifyEventT
    {
        static constexpr std::uint32_t maxBins = eventBins;

        ALPAKA_FN_ACC Event operator()(auto const&, Status& status, PhotonT<T> const& photon) const
        {
            Event const event = photon[proposalTag].event;
            if(!photon[proposalTag].valid)
            {
                status.set(StatusCode::invalid);
                return Event::absorbed;
            }
            std::uint32_t const bin = static_cast<std::uint32_t>(event);
            if(bin >= maxBins)
            {
                status.set(StatusCode::invalid);
                return Event::absorbed;
            }
            return event;
        }
    };

    using ClassifyEvent = ClassifyEventT<>;

    /** Updater: advance to the selected event and report checked failures. */
    template<typename T = double>
    struct AdvanceToEventT
    {
        SlabT<T> slab{};

        ALPAKA_FN_ACC UpdateResult operator()(auto const&, Status& status, Event event, PhotonT<T>& photon) const
        {
            UpdateResult const result = applyEvent(slab, event, photon);
            if(result == UpdateResult::invalid)
                status.set(StatusCode::invalid);
            else if(result == UpdateResult::overflow)
                status.set(StatusCode::overflow);
            return result;
        }
    };

    using AdvanceToEvent = AdvanceToEventT<>;

    /** Transform: change direction and increment the completed-scatter counter exactly once. */
    template<typename TTrace = NoTrace, typename T = double>
    struct ScatterT
    {
        SlabT<T> slab{};
        ScatterMode mode = ScatterMode::forward;
        mutable TTrace trace;

        ALPAKA_FN_ACC void operator()(auto const&, rng::Engine& engine, Status& status, PhotonT<T>& photon) const
        {
            if(!photon[proposalTag].valid || photon[proposalTag].event != Event::scatter)
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(mode == ScatterMode::isotropic)
                photon[directionTag] = isotropicDirection<T>(engine);
            if(!validatePhoton(slab, photon))
            {
                status.set(StatusCode::invalid);
                return;
            }
            if(photon[scatterCountTag] == std::numeric_limits<std::uint32_t>::max())
            {
                status.set(StatusCode::overflow);
                return;
            }
            trace.record(
                photon[idTag],
                TraceKind::scatter,
                photon[scatterCountTag] + 1u,
                photon[positionTag],
                photon[directionTag]);
            ++photon[scatterCountTag];
        }
    };

    using Scatter = ScatterT<>;

    /** Shared terminal sink: record one final state, trace vertex and per-id delivery count. */
    template<typename TRecords, typename TVisits, typename TTrace = NoTrace, typename T = double>
    struct RecordTerminal
    {
        mutable TRecords records;
        mutable TVisits visits;
        mutable TTrace trace;

        ALPAKA_FN_ACC void operator()(auto const& acc, Status& status, PhotonT<T> const& photon) const
        {
            if(!photon[proposalTag].valid)
            {
                status.set(StatusCode::invalid);
                return;
            }
            Event const event = photon[proposalTag].event;
            std::uint32_t const bin = static_cast<std::uint32_t>(event);
            if(bin == static_cast<std::uint32_t>(Event::scatter) || bin >= eventBins)
            {
                status.set(StatusCode::invalid);
                return;
            }

            records[photon[idTag]]
                = TerminalRecordT<T>{event, photon[scatterCountTag], photon[positionTag], photon[directionTag]};
            trace.record(
                photon[idTag],
                traceKindOf(event),
                photon[scatterCountTag],
                photon[positionTag],
                photon[directionTag]);
            std::uint32_t const previous = alpaka::onAcc::atomicAdd(
                acc,
                &visits[photon[idTag]],
                std::uint32_t{1u},
                alpaka::onAcc::scope::device);
            if(previous != 0u)
                status.set(StatusCode::invalid);
        }
    };
} // namespace graphFlow::photon
