/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
    /** Select the accelerator compiled into this build, or the CPU backend otherwise.
     *
     * A CUDA/HIP build must have a visible device: the test throws instead of silently falling
     * back to the host backend, so a device-less accelerator run fails rather than skips.
     */
    auto makeNativeTestDevice()
    {
#if ALPAKA_LANG_CUDA
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"CUDA build requires a visible NVIDIA GPU"};
        return selector.makeDevice(0u);
#elif ALPAKA_LANG_HIP
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{alpaka::api::hip, alpaka::deviceKind::amdGpu});
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"HIP build requires a visible AMD GPU"};
        return selector.makeDevice(0u);
#else
        return graphFlow::test::makeCpuDevice();
#endif
    }

    std::uint64_t countZeroWeights(std::vector<std::uint32_t> const& weights)
    {
        return static_cast<std::uint64_t>(std::count(weights.begin(), weights.end(), std::uint32_t{0u}));
    }

    void checkAgainstReference(
        graphFlow::particle::Parameters const& parameters,
        graphFlow::particle::FeedbackResult const& result)
    {
        graphFlow::test::ScalarOutcome const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

        // Histogram multiset equality is the acceptance contract.
        CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));

        std::uint64_t countSum = 0u;
        for(std::size_t i = 0u; i < result.histogram.size(); ++i)
        {
            countSum += result.histogram[i].count;
            if(i > 0u)
                CHECK(result.histogram[i - 1u].weight < result.histogram[i].weight);
        }
        CHECK(countSum == parameters.numParticles);

        // Element-wise equality is the diagnostic check for missing or duplicated ids: the sink
        // writes the result buffer by id, so a lost or doubled particle would surface here.
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);
        CHECK(result.cappedParticles == reference.cappedParticles);

        // The sentinel would survive any particle that never reached the sink.
        for(std::uint32_t const weight : result.weights)
            CHECK(weight != 0xDEAD'BEEFu);
    }
} // namespace

TEST_CASE("particle feedback matches the scalar oracle for the N matrix", "[integration]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    for(std::uint64_t const particles : {0ull, 1ull, 17ull, 128ull, 4097ull})
    {
        for(std::uint64_t const seed : {0ull, 7ull, 42ull})
        {
            graphFlow::particle::Parameters parameters{};
            parameters.numParticles = particles;
            parameters.chunkSize = 128u;
            parameters.seed = seed;
            parameters.maxVisits = 64u;

            CAPTURE(particles, seed, parameters.chunkSize);
            auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
            checkAgainstReference(parameters, result);
        }
    }
}

TEST_CASE("slot arena produces the same physics as the fixed layout", "[integration][arena]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    auto const fixed = graphFlow::particle::runParticleFeedback(device, parameters);
    parameters.poolMode = graphFlow::PoolMode::slotArena;
    auto const arena = graphFlow::particle::runParticleFeedback(device, parameters);

    // Same graph, same seed, same admission: only the chunk-slot layout and the base lookup
    // changed, so both runs must produce the same ids, weights and histogram.
    CHECK(arena.weights == fixed.weights);
    CHECK(arena.histogram == fixed.histogram);
    CHECK(arena.cappedParticles == fixed.cappedParticles);
    CHECK(arena.sunkParticles == fixed.sunkParticles);
    CHECK(arena.liveParticles == fixed.liveParticles);
    // The chunk-index count is P + K*R in both layouts; only the slot count differs.
    CHECK(arena.stats.poolChunks == fixed.stats.poolChunks);
    checkAgainstReference(parameters, arena);
}

TEST_CASE("occupancy target does not change particle feedback physics", "[integration][occupancy]")
{
    auto const device = makeNativeTestDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    // The feedback graph registers six nodes, so eight is a reachable target that still bounds a
    // 33-chunk source stream and exercises the admission gate.
    graphFlow::Tuning tuning = graphFlow::particle::tuningOf(parameters);
    tuning.occupancyTargetChunks = 8u;

    auto const baseline = graphFlow::particle::runParticleFeedback(device, parameters);
    auto const gated = graphFlow::particle::runParticleFeedback(device, parameters, tuning);

    CHECK(gated.stats.occupancyTargetChunks == 8u);
    CHECK(gated.stats.sourceSunk == parameters.numParticles);
    CHECK(gated.stats.liveWorkChunks == 0u);
    // The gate's engagement is asserted by the admission tests on their live-work record.
    // `stats.refills` counts admissions taken while the refill flag happened to be pending, which
    // depends on scheduler interleaving, so it stays a reported statistic rather than an
    // assertion and this case keeps its focus on the physics comparison below.
    // Scheduling changed, physics did not.
    CHECK(gated.weights == baseline.weights);
    CHECK(gated.histogram == baseline.histogram);
    CHECK(gated.cappedParticles == baseline.cappedParticles);
    checkAgainstReference(parameters, gated);
}

TEST_CASE("particle feedback matches the scalar oracle for the capacity matrix", "[integration]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    for(std::uint64_t const particles : {1ull, 17ull, 128ull, 4097ull})
    {
        for(std::uint32_t const capacity : {1u, 31u, 127u, 128u})
        {
            for(std::uint64_t const seed : {0ull, 7ull, 42ull})
            {
                graphFlow::particle::Parameters parameters{};
                parameters.numParticles = particles;
                parameters.chunkSize = capacity;
                parameters.seed = seed;
                parameters.maxVisits = 64u;

                CAPTURE(particles, capacity, seed);
                auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
                checkAgainstReference(parameters, result);
            }
        }
    }
}

TEST_CASE("particle feedback matches the scalar oracle with chunks larger than T", "[integration]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    // C=1024 with T=128 gives eight blocks per full chunk; N=1000 ends in a partial tail tile
    // inside the only chunk, N=4097 exercises full chunks plus a one-particle tail chunk.
    for(std::uint64_t const particles : {1000ull, 4097ull})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = particles;
        parameters.chunkSize = 1024u;
        parameters.tileSize = 128u;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;

        CAPTURE(particles, parameters.chunkSize, parameters.tileSize);
        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        checkAgainstReference(parameters, result);
    }
}

TEST_CASE("max-visits guard terminates pathological feedback", "[integration]")
{
    auto const device = graphFlow::test::makeCpuDevice();

    for(std::uint32_t const maxVisits : {0u, 1u, 2u})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = 512u;
        parameters.chunkSize = 31u;
        parameters.seed = 42u;
        parameters.maxVisits = maxVisits;

        CAPTURE(maxVisits);
        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        checkAgainstReference(parameters, result);
        CHECK(result.sunkParticles == parameters.numParticles);
        // A particle can never accumulate more than maxVisits feedback increments.
        for(std::uint32_t const weight : result.weights)
            CHECK(weight <= maxVisits * 1000u);
    }
}

TEST_CASE("particle feedback example is a public Flow graph with a derived pool", "[integration][flow]")
{
    using graphFlow::particle::ParticleRecord;
    static_assert(graphFlow::SourceFor<graphFlow::particle::GenerateParticles, ParticleRecord>);
    static_assert(graphFlow::PartitionClassifierFor<graphFlow::particle::FilterByProperty, ParticleRecord>);
    static_assert(graphFlow::PartitionClassifierFor<graphFlow::particle::SelectOutcome, ParticleRecord>);
    static_assert(
        graphFlow::UpdateFor<graphFlow::particle::UpdateWeight, graphFlow::particle::Outcome, ParticleRecord>);
    static_assert(graphFlow::AttributeRecord<ParticleRecord>);

    auto const device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    static_assert(std::same_as<
                  typename graphFlow::SoAFlow<ParticleRecord, Device>::Storage,
                  graphFlow::particle::ParticleSoAStorage<Device>>);

    // A subset proxy narrows access only; loading it still copies every attribute.
    using WeightSubset = graphFlow::Set<graphFlow::particle::WeightTag>;
    ParticleRecord original{7u, 3u, 11u, 2u};
    auto proxy = original.proxy<WeightSubset>();
    proxy[graphFlow::particle::weightTag] = 21u;
    ParticleRecord const copied = proxy.load();
    CHECK(copied[graphFlow::particle::idTag] == 7u);
    CHECK(copied[graphFlow::particle::propertyTag] == 3u);
    CHECK(copied[graphFlow::particle::weightTag] == 21u);
    CHECK(copied[graphFlow::particle::visitTag] == 2u);

    // The Record specialization must key the RNG by the logical id and visit count.
    CHECK(graphFlow::rng::streamId(original) == original[graphFlow::particle::idTag]);
    CHECK(graphFlow::rng::streamEvent(original) == original[graphFlow::particle::visitTag]);
    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
    graphFlow::test::ScalarOutcome const reference
        = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

    CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
    CHECK(result.weights == reference.weights);
    CHECK(result.sunkParticles == parameters.numParticles);
    CHECK(result.cappedParticles == reference.cappedParticles);
    CHECK(result.liveParticles == 256u);
    // Flow sums the registered output bounds: filter 2 + A 3 + B 3 = 8 reserve chunks.
    CHECK(result.stats.poolChunks == 256u + 8u);
}

TEST_CASE("max-visits cap bounds non-terminating feedback", "[integration][max-visits]")
{
    auto const device = makeNativeTestDevice();
    constexpr std::uint64_t particles = 1024u;

    for(std::uint32_t const maxVisits : {0u, 1u, 2u, 3u, 7u, 64u})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = particles;
        parameters.chunkSize = 128u;
        parameters.liveParticles = 256u;
        parameters.seed = 42u;
        parameters.maxVisits = maxVisits;

        CAPTURE(maxVisits);
        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        graphFlow::test::ScalarOutcome const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

        checkAgainstReference(parameters, result);
        CHECK(result.cappedParticles == reference.cappedParticles);
        // A capped particle stops at visit == maxVisits, so it accumulated at most that many
        // feedback increments. The weakest increment is 1 and the strongest 1000.
        for(std::uint32_t const weight : result.weights)
            CHECK(weight <= maxVisits * 1000u);

        // The guard is live, not an unused parameter: the oracle caps particles for every
        // bounded run in this matrix and never caps at the default 64 visits.
        if(maxVisits < 64u)
            CHECK(reference.cappedParticles > 0u);
        else
            CHECK(reference.cappedParticles == 0u);

        if(maxVisits == 0u)
        {
            // No feedback increment is allowed, so every particle - capped or sunk by the
            // survival draw - ends with weight zero in one histogram row.
            REQUIRE_FALSE(result.histogram.empty());
            CHECK(result.histogram.size() == 1u);
            CHECK(result.histogram.front().weight == 0u);
            CHECK(result.histogram.front().count == particles);
            for(std::uint32_t const weight : result.weights)
                CHECK(weight == 0u);
        }
    }
}

TEST_CASE("zero-weight particles are sunk and form the first histogram row", "[integration][zero-weight]")
{
    auto const device = makeNativeTestDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 512u;
    parameters.chunkSize = 31u;
    parameters.liveParticles = 128u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
    graphFlow::test::ScalarOutcome const reference
        = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

    checkAgainstReference(parameters, result);

    std::uint64_t const zeroCount = countZeroWeights(reference.weights);
    REQUIRE(zeroCount > 0u);
    REQUIRE(zeroCount < parameters.numParticles);
    REQUIRE_FALSE(result.histogram.empty());
    CHECK(result.histogram.front().weight == 0u);
    CHECK(result.histogram.front().count == zeroCount);
    // Zero is a real weight, not the sink sentinel: every id reached the sink.
    for(std::uint32_t const weight : result.weights)
        CHECK(weight != graphFlow::particle::missingWeight);
}

TEST_CASE("empty particle input completes without launching work", "[integration][empty]")
{
    auto const device = makeNativeTestDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 0u;
    parameters.chunkSize = 128u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    auto const result = graphFlow::particle::runParticleFeedback(device, parameters);

    CHECK(result.weights.empty());
    CHECK(result.histogram.empty());
    CHECK(result.sunkParticles == 0u);
    CHECK(result.cappedParticles == 0u);
    // The empty run keeps the historical one-particle admission header.
    CHECK(result.liveParticles == 1u);

    CHECK(result.stats.poolChunks == 1u);
    CHECK(result.stats.peakLiveChunks == 0u);
    CHECK(result.stats.peakReservedChunks == 0u);
    CHECK(result.stats.completedBatches == 0u);
    CHECK(result.stats.sourceParticles == 0u);
    CHECK(result.stats.sourceGenerated == 0u);
    CHECK(result.stats.sourceSunk == 0u);
    CHECK(result.stats.sourceLiveParticles == 0u);
    CHECK(result.stats.chunkCapacity == parameters.chunkSize);
}

TEST_CASE("exact chunk and admission fill conserves every particle", "[integration][exact-fill]")
{
    auto const device = makeNativeTestDevice();

    auto const verify = [&](std::uint64_t particles, std::uint32_t capacity, std::uint32_t live)
    {
        CAPTURE(particles, capacity, live);
        REQUIRE(particles % capacity == 0u);
        REQUIRE(particles % live == 0u);

        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = particles;
        parameters.chunkSize = capacity;
        parameters.liveParticles = live;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;

        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        checkAgainstReference(parameters, result);

        CHECK(result.liveParticles == live);
        CHECK(result.stats.poolChunks == live + 8u);
        CHECK(result.stats.peakLiveChunks <= result.stats.poolChunks);
        CHECK(result.stats.peakReservedChunks <= result.stats.poolChunks);
        CHECK(result.stats.sourceParticles == particles);
        CHECK(result.stats.sourceGenerated == particles);
        CHECK(result.stats.sourceSunk == particles);
        CHECK(result.stats.sourceLiveParticles == 0u);
        CHECK(result.stats.chunkCapacity == capacity);
        CHECK(result.stats.completedBatches > 0u);
    };

    // One full admission round: two full C-sized source chunks exactly fill P.
    verify(256u, 128u, 256u);
    // Several whole admission rounds: eight full C-sized chunks in four live batches of 2C.
    verify(1024u, 128u, 256u);
}

TEST_CASE("slot arena completes the N=100000 C=8192 run with the fixed physics", "[integration][arena][stress]")
{
    // Regression for the confirmed `main` failure: the scatter output needs a contiguous
    // C-slot range and the first-fit free list was fragmented by the feedback stream exactly at
    // this geometry. The run must complete, and only the layout may differ.
    auto const device = graphFlow::test::makeCpuDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 100000u;
    parameters.chunkSize = 8192u;
    parameters.liveParticles = 16384u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    // The fixed layout needs P*C slots, so the shared policy budget is 4 GiB here. The arena run
    // uses the same P and C but a small fraction of the slots, which is the whole point of the
    // class.
    graphFlow::Tuning fixedTuning = graphFlow::particle::tuningOf(parameters);
    fixedTuning.policyMemoryBudgetBytes = 4ull << 30u;
    fixedTuning.poolPayloadBudgetBytes = 16ull << 30u;
    auto const fixed = graphFlow::particle::runParticleFeedback(device, parameters, fixedTuning);
    checkAgainstReference(parameters, fixed);

    // The feedback graph has one B=2 and two B=3 partitions, so the output-credit reserve is
    // outputChunkBound(C,C,2) + 2*outputChunkBound(C,C,3) = 8 chunks.
    std::uint64_t const reserve = graphFlow::outputChunkBound(parameters.chunkSize, parameters.chunkSize, 2u)
                                  + 2u * graphFlow::outputChunkBound(parameters.chunkSize, parameters.chunkSize, 3u);

    // The arena budget cannot be a literal. The number of live full-capacity output ranges is a
    // scheduling property of the host, not a geometry constant: a 64 MiB arena holds 339 full
    // chunks at this geometry and failed in CI when the run reached a 340th live range, while the
    // same tree needed 193 to 240 ranges on quieter hosts. Calibrate once under the provable
    // worst-case reservation (one full-capacity range per chunk index, `P + K*R`) and size the
    // measured arena from the requirement that run reports. The peak still moves with host load,
    // so the requirement is tripled: the largest observed swing is 340 against 193 (1.76x), which
    // leaves the sized arena far above any peak seen so far and still far below the fixed pool.
    graphFlow::Tuning calibration = fixedTuning;
    calibration.poolMode = graphFlow::PoolMode::slotArena;
    calibration.slotArenaBudgetBytes = graphFlow::requiredArenaBytes(
        parameters.liveParticles,
        parameters.chunkSize,
        parameters.liveParticles + reserve);
    auto const measured = graphFlow::particle::runParticleFeedback(device, parameters, calibration);
    CHECK(measured.weights == fixed.weights);

    std::uint64_t const sizedSlots = 3u * measured.stats.requiredArenaSlots;
    graphFlow::Tuning arenaTuning = fixedTuning;
    arenaTuning.poolMode = graphFlow::PoolMode::slotArena;
    arenaTuning.slotArenaBudgetBytes = sizedSlots * graphFlow::payloadBytesPerSlot;
    auto const arena = graphFlow::particle::runParticleFeedback(device, parameters, arenaTuning);

    CHECK(arena.weights == fixed.weights);
    CHECK(arena.histogram == fixed.histogram);
    CHECK(arena.cappedParticles == fixed.cappedParticles);
    CHECK(arena.sunkParticles == fixed.sunkParticles);
    CHECK(arena.stats.poolChunks == fixed.stats.poolChunks);
    CHECK(arena.stats.sourceSunk == parameters.numParticles);
    CHECK(arena.stats.poolSlots == sizedSlots);
    CHECK(arena.stats.poolBytes == sizedSlots * graphFlow::payloadBytesPerSlot);
    // Sufficiency: the derived arena serves this execution, so an under-stated derivation throws
    // instead of retrying.
    CHECK(arena.stats.requiredArenaSlots <= arena.stats.poolSlots);
    // The peak live output chunks, not the K*R credits, size the full-chunk class.
    CHECK(arena.stats.peakLiveFullChunks > 0u);
    // The full-chunk class grows from the tail down to the general high-water mark, so the arena
    // fits `(poolSlots - P) / C` full ranges, not `poolSlots / C`. The general P slots are
    // permanently below that mark; a bound that divides the whole arena by C is two chunks loose
    // here and accepts a size the allocator rejects.
    CHECK(arena.stats.peakLiveFullChunks <= (arena.stats.poolSlots - parameters.liveParticles) / parameters.chunkSize);
    // The memory claim is stated against the fixed pool at the same P and C, not against a
    // literal arena size: the derived arena keeps an order of magnitude more headroom.
    CHECK(arena.stats.poolSlots * 10u < fixed.stats.poolSlots);
    checkAgainstReference(parameters, arena);
}

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
TEST_CASE("slot arena completes the N=100000 C=8192 run on the native device", "[integration][arena][stress]")
{
    auto const device = makeNativeTestDevice();

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 100000u;
    parameters.chunkSize = 8192u;
    parameters.liveParticles = 16384u;
    parameters.seed = 42u;
    parameters.maxVisits = 64u;

    graphFlow::Tuning tuning = graphFlow::particle::tuningOf(parameters);
    tuning.policyMemoryBudgetBytes = 4ull << 30u;
    tuning.poolPayloadBudgetBytes = 16ull << 30u;
    auto const fixed = graphFlow::particle::runParticleFeedback(device, parameters, tuning);

    tuning.poolMode = graphFlow::PoolMode::slotArena;
    tuning.slotArenaBudgetBytes = 64ull << 20u;
    auto const arena = graphFlow::particle::runParticleFeedback(device, parameters, tuning);
    CHECK(arena.weights == fixed.weights);
    CHECK(arena.histogram == fixed.histogram);
    CHECK(arena.stats.poolChunks == fixed.stats.poolChunks);
    CHECK(arena.stats.sourceSunk == parameters.numParticles);
}
#endif

TEST_CASE("the slot-arena budget knob reproduces the memory reduction", "[integration][arena][budget]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    constexpr std::uint32_t capacity = 4224u;
    constexpr std::uint32_t liveParticles = 8448u;

    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 20000u;
    parameters.chunkSize = capacity;
    parameters.liveParticles = liveParticles;
    parameters.seed = 42u;
    parameters.maxVisits = 8u;

    // The host policy budget is 16 MiB, which cannot hold the fixed pool of this configuration;
    // the probe states its budget explicitly so both layouts are measured under the same one.
    graphFlow::Tuning base = graphFlow::particle::tuningOf(parameters);
    base.policyMemoryBudgetBytes = 1ull << 30u;
    base.poolPayloadBudgetBytes = 1ull << 30u;
    auto const fixed = graphFlow::particle::runParticleFeedback(device, parameters, base);

    // The feedback graph has one B=2 and two B=3 partitions, so `transformReserve` is
    // outputChunkBound(C,C,2) + 2*outputChunkBound(C,C,3) = 2 + 6 = 8 chunks.
    std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, 2u)
                                  + 2u * graphFlow::outputChunkBound(capacity, capacity, 3u);
    CHECK(reserve == 8u);

    // Why `K*R` is not the arena requirement: the credit is released as soon as the scatter is
    // planned, while the output range stays live until a sink (or a downstream transform)
    // releases the chunk. The number of live full-capacity output ranges is therefore a
    // scheduling and queueing property of the run, not the credit count. Calibrate it with one
    // run whose arena covers the provable worst case (one full-capacity range per chunk index,
    // `S = P + K*R`), and let the run report what it actually needed.
    graphFlow::Tuning calibration = base;
    calibration.poolMode = graphFlow::PoolMode::slotArena;
    calibration.slotArenaBudgetBytes = graphFlow::requiredArenaBytes(liveParticles, capacity, liveParticles + reserve);
    auto const measured = graphFlow::particle::runParticleFeedback(device, parameters, calibration);
    CHECK(measured.weights == fixed.weights);

    // The run reports `requiredArenaSlots = peakGeneralHighWater + peakLiveFullChunks * C`, both
    // of them the run's own maxima. Their sum is a sufficient size, not a minimal one: the two
    // maxima were not necessarily reached at the same time, so it is an upper bound on
    // `max_t(generalHigh(t) + liveFull(t) * C)`.
    std::uint64_t const measuredRanges = measured.stats.peakLiveFullChunks;
    std::uint64_t const measuredSlots = measured.stats.requiredArenaSlots;
    std::uint64_t const generalSlots = measuredSlots - measuredRanges * capacity;
    CHECK(measuredRanges > reserve);
    CHECK(measuredSlots > graphFlow::requiredArenaSlots(liveParticles, capacity, reserve));
    CHECK(generalSlots >= liveParticles);

    // The requirement formula, derived independently from the public `P + R * C` reservation and
    // the calibration run's live-output peak. The report must equal it: a derivation that adds a
    // constant, uses a different peak or drops the general term makes this comparison fail, so
    // an over-stated or under-stated requirement cannot reach the sizing step.
    std::uint64_t const formulaSlots = graphFlow::requiredArenaSlots(liveParticles, capacity, measuredRanges);
    CHECK(formulaSlots == measuredSlots);

    // Size the arena from that computed requirement only. The live-output peak is a scheduling
    // property that moves with host thread scheduling, so a second probe at the same worst-case
    // arena measures the spread under the current load, and one stated floor of `8 * reserve`
    // full-chunk ranges covers a quiet host. Neither term is a retry: both are applied once,
    // before the single sized run, and there is no grow-until-it-fits loop.
    graphFlow::Tuning spreadProbe = calibration;
    auto const probed = graphFlow::particle::runParticleFeedback(device, parameters, spreadProbe);
    std::uint64_t const spreadSlots = probed.stats.requiredArenaSlots > formulaSlots
                                          ? probed.stats.requiredArenaSlots - formulaSlots
                                          : formulaSlots - probed.stats.requiredArenaSlots;
    std::uint64_t const sizedSlots = formulaSlots + 8u * reserve * capacity + 4u * spreadSlots;

    graphFlow::Tuning tuned = base;
    tuned.poolMode = graphFlow::PoolMode::slotArena;
    tuned.slotArenaBudgetBytes = sizedSlots * graphFlow::payloadBytesPerSlot;
    auto const arena = graphFlow::particle::runParticleFeedback(device, parameters, tuned);

    CHECK(arena.weights == fixed.weights);
    CHECK(arena.histogram == fixed.histogram);
    CHECK(arena.stats.poolSlots == sizedSlots);
    // Sufficiency: this execution needs no more slots than the formula-sized arena provides. An
    // under-stated formula fails here because the single sized run throws instead of retrying.
    CHECK(arena.stats.requiredArenaSlots <= arena.stats.poolSlots);

    // Layout accounting: the fixed pool charges one C-row per chunk index, while the arena
    // charges one slot per live particle plus one C range per concurrent output chunk.
    CHECK(fixed.stats.poolChunks == liveParticles + reserve);
    CHECK(fixed.stats.poolSlots == static_cast<std::uint64_t>(fixed.stats.poolChunks) * capacity);
    CHECK(fixed.stats.poolBytes == fixed.stats.poolSlots * graphFlow::payloadBytesPerSlot);
    CHECK(arena.stats.poolChunks == fixed.stats.poolChunks);
    // The memory claim is stated against the requirement the run reported, not against the
    // probe margin, so an over-stated formula cannot hide behind a larger arena. The factor is
    // conservative because the live-output peak, and therefore the requirement, grows when the
    // host is loaded; the measured ratio for this configuration is reported by the probe.
    CHECK(formulaSlots * 10u < static_cast<std::uint64_t>(fixed.stats.poolChunks) * capacity);
    // The depth stays bounded by the same budget: the reservation inversion returns exactly the
    // live-particle budget when it is evaluated at the requirement's peak.
    CHECK(
        graphFlow::maxLiveParticlesForArena(formulaSlots * graphFlow::payloadBytesPerSlot, capacity, measuredRanges)
        == liveParticles);
}

TEST_CASE("an undersized slot-arena budget is rejected before the first launch", "[integration][arena][budget]")
{
    auto const device = graphFlow::test::makeCpuDevice();
    graphFlow::particle::Parameters parameters{};
    parameters.numParticles = 4097u;
    parameters.chunkSize = 128u;
    parameters.liveParticles = 256u;
    parameters.seed = 42u;
    parameters.maxVisits = 8u;

    graphFlow::Tuning tuning = graphFlow::particle::tuningOf(parameters);
    tuning.poolMode = graphFlow::PoolMode::slotArena;
    tuning.slotArenaBudgetBytes = 1024u;

    std::string message;
    try
    {
        [[maybe_unused]] auto const result = graphFlow::particle::runParticleFeedback(device, parameters, tuning);
    }
    catch(std::invalid_argument const& error)
    {
        message = error.what();
    }
    // `Config::validate()` runs in `resolveConfig`, before the storage is built and before the
    // first kernel launch, and names both the required and the available slot count.
    CHECK(message.find("needs 1280 slots") != std::string::npos);
    CHECK(message.find("but has 42") != std::string::npos);
}
