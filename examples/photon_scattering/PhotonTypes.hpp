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
    /** Attribute tags of the AoS-over-SoA photon payload.
     *
     * The payload is a `graphFlow::Record` over these fields, so a stage reads and writes one
     * logical photon through `photon[idTag]`, `photon[positionTag]`, and so on while the storage
     * keeps one column per attribute. The tags carry no runtime state.
     */
    GRAPHFLOW_TAG(Id, idTag);
    GRAPHFLOW_TAG(Position, positionTag);
    GRAPHFLOW_TAG(Direction, directionTag);
    GRAPHFLOW_TAG(ScatterCount, scatterCountTag);
    GRAPHFLOW_TAG(Proposal, proposalTag);

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

    /** Three-dimensional vector in the photon scalar type. */
    template<typename T>
    struct Vec3T
    {
        T x = T{0};
        T y = T{0};
        T z = T{0};

        friend constexpr bool operator==(Vec3T const&, Vec3T const&) = default;
    };

    /** Double-precision photon vector used by the CPU, CUDA and HIP backends. */
    using Vec3 = Vec3T<double>;

    /** Homogeneous plane-parallel slab in inverse-length units. */
    template<typename T>
    struct SlabT
    {
        T thickness = T{1};
        T sigmaS = T{0};
        T sigmaA = T{0};
    };

    using Slab = SlabT<double>;

    /** Flight proposal carried inside the payload.
     *
     * `event` and `distance` are the checked result of the propose stage. `valid` is false when
     * the input geometry is non-finite or impossible; the partition then rejects the item through
     * `Status` before publishing any output.
     */
    template<typename T>
    struct ProposalT
    {
        Event event = Event::absorbed;
        T distance = T{0};
        bool valid = false;
    };

    using Proposal = ProposalT<double>;

    /** Attribute descriptors of the photon record.
     *
     * The field pack is the single source of truth for the `Record` payload and its `SoAStorage`
     * columns. Keeping the vector and proposal values as whole attributes preserves the scalar
     * template for both the double and single-precision paths.
     */
    using PhotonIdField = graphFlow::Field<IdTag, std::uint64_t>;

    template<typename T>
    using PhotonPositionField = graphFlow::Field<PositionTag, Vec3T<T>>;

    template<typename T>
    using PhotonDirectionField = graphFlow::Field<DirectionTag, Vec3T<T>>;

    using PhotonScatterCountField = graphFlow::Field<ScatterCountTag, std::uint32_t>;

    template<typename T>
    using PhotonProposalField = graphFlow::Field<ProposalTag, ProposalT<T>>;

    /** Photon payload of the public `Flow` API.
     *
     * The record stores one logical photon while the selected storage keeps a separate column for
     * each attribute. `scatterCount` counts completed scatter interactions and is also the stream
     * event index, so every flight decision, collision-kind decision and angular draw depends only
     * on the packed `(seed, id, scatterCount, purpose)` engine position.
     */
    template<typename T>
    using PhotonT = graphFlow::Record<
        PhotonIdField,
        PhotonPositionField<T>,
        PhotonDirectionField<T>,
        PhotonScatterCountField,
        PhotonProposalField<T>>;

    using Photon = PhotonT<double>;

    /** AoS-over-SoA storage matching the photon record attribute pack. */
    template<typename TDevice, typename T = double>
    using PhotonStorage = graphFlow::SoAStorage<
        TDevice,
        PhotonIdField,
        PhotonPositionField<T>,
        PhotonDirectionField<T>,
        PhotonScatterCountField,
        PhotonProposalField<T>>;

    static_assert(std::is_trivially_copyable_v<Photon>);
    static_assert(std::is_default_constructible_v<Photon>);

    /** Structured SIMD pack of the photon payload.
     *
     * One pack holds `T_width` photons. The three-vector attributes keep three packs each and are
     * combined componentwise, which is the SIMD idiom the alpaka `HACCmk` benchmark uses. The
     * decision fields live in the scalar domain the arithmetic uses: `event` and `scatterCount`
     * are integral values held in a floating-point pack, and `valid` is `1` or `0`. Every value is
     * exactly representable, so a lane converts back to the payload bit for bit. Keeping the whole
     * state in one pack lets every selection be a comparison mask over the same value type.
     */
    template<typename T, std::uint32_t T_width>
    struct PhotonSimdPack
    {
        static constexpr std::uint32_t width = T_width;

        using Scalar = T;

        alpaka::Simd<std::uint64_t, T_width> id;
        alpaka::Simd<T, T_width> positionX;
        alpaka::Simd<T, T_width> positionY;
        alpaka::Simd<T, T_width> positionZ;
        alpaka::Simd<T, T_width> directionX;
        alpaka::Simd<T, T_width> directionY;
        alpaka::Simd<T, T_width> directionZ;
        alpaka::Simd<T, T_width> scatterCount;
        alpaka::Simd<T, T_width> event; // proposal event as an integral value
        alpaka::Simd<T, T_width> distance;
        alpaka::Simd<T, T_width> valid; // proposal valid as 1 or 0
    };
} // namespace graphFlow::photon

namespace graphFlow
{
    /** SIMD form of the photon record.
     *
     * The storage keeps one column per attribute, so the position and direction columns hold
     * interleaved three-vectors. `load` transposes them into the three component packs of
     * `PhotonSimdPack`; this is the only place the payload layout is transposed. `lane` is the
     * inverse for the scattered output store.
     */
    template<std::uint32_t T_width, typename T>
    struct SimdPayload<photon::PhotonT<T>, T_width>
    {
        using Pack = photon::PhotonSimdPack<T, T_width>;
        using Simd = alpaka::Simd<T, T_width>;

        template<typename... TColumnPacks>
        ALPAKA_FN_ACC static Pack load(std::tuple<TColumnPacks...> const& columns)
        {
            static_assert(sizeof...(TColumnPacks) == 5u, "graphFlow: photon payload expects five columns");
            auto const& ids = std::get<0>(columns);
            auto const& positions = std::get<1>(columns);
            auto const& directions = std::get<2>(columns);
            auto const& counts = std::get<3>(columns);
            auto const& proposals = std::get<4>(columns);

            Pack pack;
            pack.id = ids;
            pack.positionX = Simd{[&positions](std::uint32_t const lane) { return positions[lane].x; }};
            pack.positionY = Simd{[&positions](std::uint32_t const lane) { return positions[lane].y; }};
            pack.positionZ = Simd{[&positions](std::uint32_t const lane) { return positions[lane].z; }};
            pack.directionX = Simd{[&directions](std::uint32_t const lane) { return directions[lane].x; }};
            pack.directionY = Simd{[&directions](std::uint32_t const lane) { return directions[lane].y; }};
            pack.directionZ = Simd{[&directions](std::uint32_t const lane) { return directions[lane].z; }};
            pack.scatterCount = Simd{[&counts](std::uint32_t const lane) { return static_cast<T>(counts[lane]); }};
            pack.event = Simd{[&proposals](std::uint32_t const lane)
                              { return static_cast<T>(static_cast<std::uint32_t>(proposals[lane].event)); }};
            pack.distance = Simd{[&proposals](std::uint32_t const lane) { return proposals[lane].distance; }};
            pack.valid = Simd{[&proposals](std::uint32_t const lane) { return proposals[lane].valid ? T{1} : T{0}; }};
            return pack;
        }

        ALPAKA_FN_ACC static photon::PhotonT<T> lane(Pack const& pack, std::uint32_t const lane)
        {
            photon::PhotonT<T> payload;
            payload[photon::idTag] = pack.id[lane];
            payload[photon::positionTag]
                = photon::Vec3T<T>{pack.positionX[lane], pack.positionY[lane], pack.positionZ[lane]};
            payload[photon::directionTag]
                = photon::Vec3T<T>{pack.directionX[lane], pack.directionY[lane], pack.directionZ[lane]};
            payload[photon::scatterCountTag] = static_cast<std::uint32_t>(pack.scatterCount[lane]);
            payload[photon::proposalTag] = photon::ProposalT<T>{
                static_cast<photon::Event>(static_cast<std::uint32_t>(pack.event[lane])),
                pack.distance[lane],
                pack.valid[lane] != T{0}};
            return payload;
        }
    };
} // namespace graphFlow

namespace graphFlow::rng
{
    /** RNG stream fields of the photon record.
     *
     * The generic `StreamTraits` reads data members, which a `Record` does not expose. Without this
     * specialization every photon would share the `(0, 0)` stream position and therefore the same
     * history. The photon key uses the logical id and the completed scatter count, matching the
     * pre-record payload and the scalar oracle.
     */
    template<typename T>
    struct StreamTraits<photon::PhotonT<T>>
    {
        static ALPAKA_FN_ACC std::uint64_t id(photon::PhotonT<T> const& payload) noexcept
        {
            return static_cast<std::uint64_t>(payload[photon::idTag]);
        }

        static ALPAKA_FN_ACC std::uint64_t event(photon::PhotonT<T> const& payload) noexcept
        {
            return static_cast<std::uint64_t>(payload[photon::scatterCountTag]);
        }
    };
} // namespace graphFlow::rng

namespace graphFlow::photon
{
    /** Application-owned terminal record written by the shared sink. */
    template<typename T>
    struct TerminalRecordT
    {
        Event event = Event::absorbed;
        std::uint32_t scatterCount = 0u;
        Vec3T<T> position{};
        Vec3T<T> direction{};
    };

    using TerminalRecord = TerminalRecordT<double>;

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
    template<typename T>
    struct TraceVertexT
    {
        TraceKind kind = TraceKind::source;
        std::uint32_t scatterCount = 0u;
        Vec3T<T> position{};
        Vec3T<T> direction{};
    };

    using TraceVertex = TraceVertexT<double>;

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
    template<typename T>
    struct PhotonTraceT
    {
        std::uint64_t id = 0u;
        std::uint32_t count = 0u;
        bool truncated = false;
        std::vector<TraceVertexT<T>> vertices;
    };

    using PhotonTrace = PhotonTraceT<double>;

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
        /** Transform launch schedule; `off` keeps the batch-wide transform launch. */
        graphFlow::ChunkKernelSchedule chunkKernelSchedule = graphFlow::ChunkKernelSchedule::off;
        /** Kernel mapping of the batch-wide transform; `batch` keeps the historical launch. */
        graphFlow::TransformKernelMode transformKernelMode = graphFlow::TransformKernelMode::batch;
        /** Per-chunk transform queues; one keeps a single serial queue. */
        std::uint32_t chunkQueueDepth = 1u;
        /** Per-chunk latency cap in microseconds fed to the launch-size policy; zero means no cap. */
        std::uint32_t chunkLatencyCapUs = 0u;
        std::uint64_t poolPayloadBudgetBytes = 0u; // zero means the Flow default
        graphFlow::PoolMode poolMode = graphFlow::PoolMode::fixed; // chunk-slot layout
        /** Payload budget of the slot arena in bytes; zero keeps the whole pool budget. */
        std::uint64_t slotArenaBudgetBytes = 0u;
        TraceConfig trace; // bounded deterministic diagnostic trace
    };

    /** Total macroscopic cross-section, `sigmaS + sigmaA`. */
    template<typename T>
    constexpr T sigmaTotal(SlabT<T> const& slab)
    {
        return slab.sigmaS + slab.sigmaA;
    }

    /** Resolve the domain parameters into the slab every stage shares. */
    template<typename T = double>
    constexpr SlabT<T> slabOf(Parameters const& parameters)
    {
        return SlabT<T>{
            static_cast<T>(parameters.thickness),
            static_cast<T>(parameters.sigmaS),
            static_cast<T>(parameters.sigmaA)};
    }

    /** Validate a resolved slab once on the host.
     *
     * The device stages assume this check has passed. They do not repeat the thickness, coefficient
     * or total checks per event.
     */
    inline void validateSlab(Slab const& slab)
    {
        if(!std::isfinite(slab.thickness) || slab.thickness <= 0.0)
            throw std::invalid_argument{"graphFlow::photon: thickness must be finite and positive"};
        if(!std::isfinite(slab.sigmaS) || slab.sigmaS < 0.0)
            throw std::invalid_argument{"graphFlow::photon: sigmaS must be finite and non-negative"};
        if(!std::isfinite(slab.sigmaA) || slab.sigmaA < 0.0)
            throw std::invalid_argument{"graphFlow::photon: sigmaA must be finite and non-negative"};
        if(!std::isfinite(sigmaTotal(slab)))
            throw std::invalid_argument{"graphFlow::photon: sigmaS + sigmaA must be finite"};
    }

    /** Validate the physical and execution configuration before a run. */
    inline void validateParameters(Parameters const& parameters)
    {
        if(parameters.numPhotons != 0u && parameters.numPhotons - 1u > graphFlow::rng::maxStreamId)
            throw std::invalid_argument{"graphFlow::photon: photon count exceeds the RNG stream-id range"};
        if(parameters.maxScatters > graphFlow::rng::maxStreamEvent)
            throw std::invalid_argument{"graphFlow::photon: max scatters exceeds the RNG stream-event range"};
        validateSlab(slabOf(parameters));
        if(parameters.chunkSize == 0u)
            throw std::invalid_argument{"graphFlow::photon: chunkSize must be non-zero"};
        if(parameters.tileSize == 0u)
            throw std::invalid_argument{"graphFlow::photon: tileSize must be non-zero"};
        if(parameters.batchChunks == 0u)
            throw std::invalid_argument{"graphFlow::photon: batchChunks must be non-zero"};
        if(parameters.frameCap == 0u)
            throw std::invalid_argument{"graphFlow::photon: frameCap must be non-zero"};
        if(parameters.chunkKernelSchedule == graphFlow::ChunkKernelSchedule::perChunk
           && parameters.chunkQueueDepth == 0u)
            throw std::invalid_argument{"graphFlow::photon: chunkQueueDepth must be non-zero for perChunk"};
        if(parameters.chunkKernelSchedule == graphFlow::ChunkKernelSchedule::perChunk
           && parameters.transformKernelMode != graphFlow::TransformKernelMode::batch)
            throw std::invalid_argument{
                "graphFlow::photon: transformKernelMode must be batch when chunkKernelSchedule is perChunk"};
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
        tuning.seed = parameters.seed;
        tuning.chunkCapacity = parameters.chunkSize;
        if(parameters.liveParticles != 0u)
            tuning.liveParticles = parameters.liveParticles;
        tuning.tileSize = parameters.tileSize;
        tuning.batchChunks = parameters.batchChunks;
        tuning.frameCap = parameters.frameCap;
        tuning.chunkKernelSchedule = parameters.chunkKernelSchedule;
        tuning.transformKernelMode = parameters.transformKernelMode;
        tuning.chunkQueueDepth = parameters.chunkQueueDepth;
        tuning.chunkLatencyCapUs = parameters.chunkLatencyCapUs;
        if(parameters.poolPayloadBudgetBytes != 0u)
            tuning.poolPayloadBudgetBytes = parameters.poolPayloadBudgetBytes;
        tuning.poolMode = parameters.poolMode;
        if(parameters.slotArenaBudgetBytes != 0u)
            tuning.slotArenaBudgetBytes = parameters.slotArenaBudgetBytes;
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
    template<typename T>
    struct ScatteringResultT
    {
        std::vector<TerminalRecordT<T>> terminal; // final record indexed by photon id
        std::vector<std::uint32_t> terminalVisits; // terminal sink writes per id
        std::vector<OutcomeHistogramRow> histograms; // sorted by event, then scatter count
        std::uint64_t eventCounts[eventBins]{}; // terminal records per event
        std::uint64_t capped = 0u; // records with Event::capped
        std::uint64_t terminated = 0u; // records with exactly one terminal write
        std::uint64_t inconsistent = 0u; // ids without exactly one terminal write
        std::uint32_t liveParticles = 0u; // configured source-admission capacity P
        std::vector<PhotonTraceT<T>> traces; // opt-in bounded deterministic traces
        std::uint64_t traceTruncated = 0u; // selected ids whose vertex buffer overflowed
        SchedulerStats stats;
    };

    using ScatteringResult = ScatteringResultT<double>;
} // namespace graphFlow::photon
