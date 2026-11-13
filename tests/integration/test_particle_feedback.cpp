/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace
{
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
            CHECK(weight != 0xDEADBEEFu);
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
    static_assert(graphFlow::SourceFor<graphFlow::particle::GenerateParticles, graphFlow::Particle>);
    static_assert(graphFlow::PartitionClassifierFor<graphFlow::particle::FilterByProperty, graphFlow::Particle>);
    static_assert(graphFlow::PartitionClassifierFor<graphFlow::particle::SelectOutcome, graphFlow::Particle>);
    static_assert(
        graphFlow::UpdateFor<graphFlow::particle::UpdateWeight, graphFlow::particle::Outcome, graphFlow::Particle>);

    auto const device = graphFlow::test::makeCpuDevice();
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
