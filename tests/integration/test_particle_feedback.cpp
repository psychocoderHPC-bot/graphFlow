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
