/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/flow/Tuning.hpp"
#include "graphFlow/graphFlow.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace graphFlow::photon
{
    /** Terminal outcomes of one photon history.
     *
     * `scatter` is a graph edge, not a terminal record. The remaining four values feed the shared
     * terminal sink; `capped` is a diagnostic outcome and never counted as physical transmission,
     * reflection or absorption.
     */
    enum class Event : std::uint32_t
    {
        scatter = 0u,
        transmitted = 1u,
        reflected = 2u,
        absorbed = 3u,
        capped = 4u
    };

    /** Number of dense partition bins (and terminal event values). */
    inline constexpr std::uint32_t eventBins = 5u;

    /** Angular model of the elastic scattering kernel. */
    enum class ScatterMode : std::uint32_t
    {
        forward = 0u,
        isotropic = 1u
    };

    /** Three-dimensional vector in double precision. */
    struct Vec3
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        friend constexpr bool operator==(Vec3 const&, Vec3 const&) = default;
    };

    /** Homogeneous plane-parallel slab in inverse-length units. */
    struct Slab
    {
        double thickness = 1.0;
        double sigmaS = 0.0;
        double sigmaA = 0.0;
    };

    /** Flight proposal carried inside the payload.
     *
     * `event` and `distance` are the checked result of the propose stage. `valid` is false when
     * the input geometry is non-finite or impossible; the partition then rejects the item through
     * `Status` before publishing any output.
     */
    struct Proposal
    {
        Event event = Event::absorbed;
        double distance = 0.0;
        bool valid = false;
    };

    /** Photon payload of the public `Flow` API.
     *
     * `scatterCount` counts completed scatter interactions and is also the Philox visit index, so
     * every flight decision, collision-kind decision and angular draw depends only on
     * `(seed, id, scatterCount, purpose)`.
     */
    struct Photon
    {
        std::uint64_t id = 0u;
        Vec3 position{};
        Vec3 direction{0.0, 0.0, 1.0};
        std::uint32_t scatterCount = 0u;
        Proposal proposal{};
    };

    static_assert(std::is_trivially_copyable_v<Photon>);
    static_assert(std::is_default_constructible_v<Photon>);

    /** Application-owned terminal record written by the shared sink. */
    struct TerminalRecord
    {
        Event event = Event::absorbed;
        std::uint32_t scatterCount = 0u;
        Vec3 position{};
        Vec3 direction{};
    };

    static_assert(std::is_trivially_copyable_v<TerminalRecord>);
    static_assert(std::is_default_constructible_v<TerminalRecord>);

    /** Ordered diagnostic vertex kind. `source` and `scatter` are intermediate vertices. */
    enum class TraceKind : std::uint32_t
    {
        source = 0u,
        scatter = 1u,
        transmitted = 2u,
        reflected = 3u,
        absorbed = 4u,
        capped = 5u
    };

    /** One bounded trace vertex: event kind, completed scatters, position and outgoing direction. */
    struct TraceVertex
    {
        TraceKind kind = TraceKind::source;
        std::uint32_t scatterCount = 0u;
        Vec3 position{};
        Vec3 direction{};
    };

    static_assert(std::is_trivially_copyable_v<TraceVertex>);
    static_assert(std::is_default_constructible_v<TraceVertex>);

    /** Opt-in bounded trace configuration.
     *
     * The trace selects the deterministic subset `id < maxPhotons` and stores at most
     * `maxVertices` vertices per selected id. Trace truncation is independent of the simulation
     * `maxScatters` guard.
     */
    struct TraceConfig
    {
        bool enabled = false;
        std::uint32_t maxPhotons = 0u;
        std::uint32_t maxVertices = 0u;
    };

    /** Host-side trace of one selected photon. */
    struct PhotonTrace
    {
        std::uint64_t id = 0u;
        std::uint32_t count = 0u;
        bool truncated = false;
        std::vector<TraceVertex> vertices;
    };

    /** Sentinel event for a result slot that no terminal sink reached. */
    inline constexpr Event missingEvent = static_cast<Event>(0xFFFF'FFFFu);

    /** Domain knobs of the photon-scattering example.
     *
     * `chunkSize` (C), `liveParticles` (P), `tileSize` (T), `batchChunks` (Q), `frameCap` and the
     * optional pool payload budget map directly to `graphFlow::Tuning`. A zero `liveParticles`
     * keeps the device-policy default and a zero `poolPayloadBudgetBytes` keeps the Flow default.
     */
    struct Parameters
    {
        std::uint64_t numPhotons = 0u;
        std::uint32_t chunkSize = 128u;
        std::uint32_t liveParticles = 0u; // P, zero means the device-policy default
        std::uint64_t seed = 0u;
        double thickness = 1.0;
        double sigmaS = 0.0;
        double sigmaA = 0.0;
        ScatterMode mode = ScatterMode::forward;
        std::uint32_t maxScatters = 64u;
        std::uint32_t tileSize = 128u;
        std::uint32_t deviceIndex = 0u;
        std::uint32_t batchChunks = 1u; // Q, ready input chunks per batch
        std::uint32_t frameCap = graphFlow::defaultFrameCap; // blocks per multiprocessor cap
        std::uint64_t poolPayloadBudgetBytes = 0u; // zero means the Flow default
        TraceConfig trace; // bounded deterministic diagnostic trace
    };

    /** Total macroscopic cross-section, `sigmaS + sigmaA`. */
    constexpr double sigmaTotal(Slab const& slab)
    {
        return slab.sigmaS + slab.sigmaA;
    }

    /** Validate the physical and execution configuration before a run. */
    inline void validateParameters(Parameters const& parameters)
    {
        if(!std::isfinite(parameters.thickness) || parameters.thickness <= 0.0)
            throw std::invalid_argument{"graphFlow::photon: thickness must be finite and positive"};
        if(!std::isfinite(parameters.sigmaS) || parameters.sigmaS < 0.0)
            throw std::invalid_argument{"graphFlow::photon: sigmaS must be finite and non-negative"};
        if(!std::isfinite(parameters.sigmaA) || parameters.sigmaA < 0.0)
            throw std::invalid_argument{"graphFlow::photon: sigmaA must be finite and non-negative"};
        double const total = sigmaTotal(Slab{parameters.thickness, parameters.sigmaS, parameters.sigmaA});
        if(!std::isfinite(total))
            throw std::invalid_argument{"graphFlow::photon: sigmaS + sigmaA must be finite"};
        if(parameters.chunkSize == 0u)
            throw std::invalid_argument{"graphFlow::photon: chunkSize must be non-zero"};
        if(parameters.tileSize == 0u)
            throw std::invalid_argument{"graphFlow::photon: tileSize must be non-zero"};
        if(parameters.batchChunks == 0u)
            throw std::invalid_argument{"graphFlow::photon: batchChunks must be non-zero"};
        if(parameters.frameCap == 0u)
            throw std::invalid_argument{"graphFlow::photon: frameCap must be non-zero"};
        if(parameters.mode != ScatterMode::forward && parameters.mode != ScatterMode::isotropic)
            throw std::invalid_argument{"graphFlow::photon: unknown scattering mode"};
        if(parameters.trace.enabled)
        {
            if(parameters.trace.maxPhotons == 0u)
                throw std::invalid_argument{"graphFlow::photon: trace maxPhotons must be non-zero"};
            if(parameters.trace.maxVertices == 0u)
                throw std::invalid_argument{"graphFlow::photon: trace maxVertices must be non-zero"};
        }
    }

    /** Map the domain knobs to the optional execution tuning of `Flow`. */
    inline graphFlow::Tuning tuningOf(Parameters const& parameters)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = parameters.chunkSize;
        if(parameters.liveParticles != 0u)
            tuning.liveParticles = parameters.liveParticles;
        tuning.tileSize = parameters.tileSize;
        tuning.batchChunks = parameters.batchChunks;
        tuning.frameCap = parameters.frameCap;
        if(parameters.poolPayloadBudgetBytes != 0u)
            tuning.poolPayloadBudgetBytes = parameters.poolPayloadBudgetBytes;
        return tuning;
    }

    /** Scatter-count histogram row of one terminal outcome. */
    struct OutcomeHistogramRow
    {
        Event event = Event::absorbed;
        std::uint32_t scatterCount = 0u;
        std::uint64_t count = 0u;

        friend bool operator==(OutcomeHistogramRow const&, OutcomeHistogramRow const&) = default;
    };

    /** Host-side result of one photon-scattering run. */
    struct ScatteringResult
    {
        std::vector<TerminalRecord> terminal; // final record indexed by photon id
        std::vector<std::uint32_t> terminalVisits; // terminal sink writes per id
        std::vector<OutcomeHistogramRow> histograms; // sorted by event, then scatter count
        std::uint64_t eventCounts[eventBins]{}; // terminal records per event
        std::uint64_t capped = 0u; // records with Event::capped
        std::uint64_t terminated = 0u; // records with exactly one terminal write
        std::uint64_t inconsistent = 0u; // ids without exactly one terminal write
        std::uint32_t liveParticles = 0u; // configured source-admission capacity P
        std::vector<PhotonTrace> traces; // opt-in bounded deterministic traces
        std::uint64_t traceTruncated = 0u; // selected ids whose vertex buffer overflowed
        SchedulerStats stats;
    };
} // namespace graphFlow::photon
