/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

#if !ALPAKA_LANG_ONEAPI
#    error "test_particle_pressure_sycl.cpp requires a SYCL compilation (GRAPHFLOW_ENABLE_SYCL=ON)"
#endif

namespace
{
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

        // Element-wise equality is the diagnostic check for missing or duplicated ids.
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);
        CHECK(result.cappedParticles == reference.cappedParticles);

        // The sentinel would survive any particle that never reached the sink.
        for(std::uint32_t const weight : result.weights)
            CHECK(weight != graphFlow::particle::missingWeight);
    }
} // namespace

TEST_CASE("SYCL particle feedback matches the scalar oracle across max-visits", "[sycl][feedback][max-visits]")
{
    auto const device = graphFlow::test::makeSyclDevice();
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
        checkAgainstReference(parameters, result);

        graphFlow::test::ScalarOutcome const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
        CHECK(result.cappedParticles == reference.cappedParticles);
        for(std::uint32_t const weight : result.weights)
            CHECK(weight <= maxVisits * 1000u);

        if(maxVisits < 64u)
            CHECK(reference.cappedParticles > 0u);
        else
            CHECK(reference.cappedParticles == 0u);
    }
}

TEST_CASE("SYCL zero-weight particles are sunk and form the first histogram row", "[sycl][feedback][zero-weight]")
{
    auto const device = graphFlow::test::makeSyclDevice();

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

TEST_CASE("SYCL empty particle input completes without launching work", "[sycl][feedback][empty]")
{
    auto const device = graphFlow::test::makeSyclDevice();

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

TEST_CASE("SYCL exact chunk and admission fill conserves every particle", "[sycl][feedback][exact-fill]")
{
    auto const device = graphFlow::test::makeSyclDevice();

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

    verify(256u, 128u, 256u);
    verify(1024u, 128u, 256u);
}

TEST_CASE("SYCL bounded pool completes with N much larger than P", "[sycl][pressure]")
{
    auto const device = graphFlow::test::makeSyclDevice();
    constexpr std::uint32_t expectedPool = 256u + 8u;

    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        graphFlow::particle::Parameters parameters{};
        parameters.numParticles = 4097u;
        parameters.chunkSize = 128u;
        parameters.liveParticles = 256u;
        parameters.seed = 42u;
        parameters.maxVisits = 64u;
        parameters.batchChunks = batchChunks;

        CAPTURE(batchChunks);
        graphFlow::particle::FeedbackResult const result
            = graphFlow::particle::runParticleFeedback(device, parameters);

        graphFlow::test::ScalarOutcome const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);
        CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);

        // The pool is derived from P, not from N or Q: filter 2 + A 3 + B 3 = 8.
        CHECK(result.stats.poolChunks == expectedPool);
        CHECK(result.stats.peakLiveChunks <= expectedPool);
        CHECK(result.stats.peakReservedChunks <= expectedPool);
        CHECK(result.stats.peakLiveChunks < parameters.numParticles);
        CHECK(result.stats.sourceParticles == parameters.numParticles);
        CHECK(result.stats.sourceGenerated == parameters.numParticles);
        CHECK(result.stats.sourceSunk == parameters.numParticles);
        CHECK(result.stats.sourceLiveParticles == 0u);
        CHECK(result.liveParticles == 256u);
    }
}

TEST_CASE("SYCL small P below C drains a fragmented pool across Q", "[sycl][pressure][small-P]")
{
    auto const device = graphFlow::test::makeSyclDevice();

    for(std::uint32_t const liveParticles : {1u, 3u, 8u})
    {
        for(std::uint32_t const batchChunks : {1u, 8u})
        {
            graphFlow::particle::Parameters parameters{};
            parameters.numParticles = 131u;
            parameters.chunkSize = 128u;
            parameters.liveParticles = liveParticles;
            parameters.seed = 7u;
            parameters.maxVisits = 64u;
            parameters.batchChunks = batchChunks;

            CAPTURE(liveParticles, batchChunks);
            graphFlow::particle::FeedbackResult const result
                = graphFlow::particle::runParticleFeedback(device, parameters);
            graphFlow::test::ScalarOutcome const reference
                = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

            CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
            CHECK(result.weights == reference.weights);
            CHECK(result.sunkParticles == parameters.numParticles);
            CHECK(result.cappedParticles == reference.cappedParticles);
            CHECK(result.stats.poolChunks == liveParticles + 8u);
            CHECK(result.stats.peakLiveChunks <= result.stats.poolChunks);
            CHECK(result.stats.sourceGenerated == parameters.numParticles);
            CHECK(result.stats.sourceSunk == parameters.numParticles);
            CHECK(result.stats.sourceLiveParticles == 0u);
            CHECK(result.liveParticles == liveParticles);
        }
    }
}
