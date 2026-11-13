/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    using U32View
        = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<Device>(), alpaka::Vec{std::uint64_t{1u}}));

    graphFlow::Tuning tuningFor(std::uint32_t capacity, std::uint32_t tileSize, std::uint32_t batchChunks)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = tileSize;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        return tuning;
    }

    template<typename T>
    std::vector<T> readBack(Device& device, U32View const& view, std::size_t size)
    {
        std::vector<T> host(size);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    struct GenerateParticle
    {
        ALPAKA_FN_ACC graphFlow::Particle operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return graphFlow::Particle{id, static_cast<std::uint32_t>(id % 3u), 0u, 0u};
        }
    };

    struct VisitIngress
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle& particle) const
        {
            particle.visit += 1u;
            particle.weight += 1u;
        }
    };

    struct FeedbackFilter
    {
        static constexpr std::uint32_t maxBins = 2u;

        std::uint32_t visitLimit = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Particle const& particle) const
        {
            if(particle.visit >= visitLimit)
                return 0u;
            return particle.id % 2u == 0u ? 1u : 0u;
        }
    };

    template<typename TVisits, typename TCounts>
    struct RecordVisit
    {
        mutable TVisits visits;
        mutable TCounts counts;

        ALPAKA_FN_ACC void operator()(auto const& acc, graphFlow::Particle const& particle) const
        {
            visits[particle.id] = particle.visit;
            alpaka::onAcc::atomicAdd(acc, &counts[particle.id], std::uint32_t{1u}, alpaka::onAcc::scope::device);
        }
    };

    struct ParticleOutcome
    {
        std::vector<std::uint32_t> visits;
        std::vector<std::uint32_t> counts;
        graphFlow::SchedulerStats stats;
    };

    ParticleOutcome runParticleFeedback(
        std::uint64_t particles,
        graphFlow::Tuning const& tuning,
        std::uint32_t* poolChunks = nullptr)
    {
        auto device = graphFlow::test::makeCpuDevice();
        std::size_t const size = static_cast<std::size_t>(std::max<std::uint64_t>(particles, 1u));
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        auto counts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            std::vector<std::uint32_t> const zero(size, 0u);
            alpaka::onHost::memcpy(queue, visits, zero);
            alpaka::onHost::memcpy(queue, counts, zero);
        }

        graphFlow::Flow<graphFlow::Particle, Device> flow{device};
        auto source = flow.source("generate", GenerateParticle{});
        auto ingress = flow.transform("ingress", VisitIngress{});
        auto filter = flow.partition("filter", FeedbackFilter{2u});
        using VisitView = decltype(visits.getView());
        using CountView = decltype(counts.getView());
        auto sink = flow.sink("record", RecordVisit<VisitView, CountView>{visits.getView(), counts.getView()});
        flow.connect(source, ingress);
        flow.connect(ingress, filter);
        flow.connect(filter.out<0u>(), sink);
        flow.connect(filter.out<1u>(), ingress);
        flow.configure(tuning);

        ParticleOutcome outcome;
        outcome.stats = flow.run(source, particles);
        if(poolChunks != nullptr)
            *poolChunks = outcome.stats.poolChunks;
        outcome.visits = readBack<std::uint32_t>(device, visits, size);
        outcome.counts = readBack<std::uint32_t>(device, counts, size);
        outcome.visits.resize(static_cast<std::size_t>(particles));
        outcome.counts.resize(static_cast<std::size_t>(particles));
        return outcome;
    }

    struct Pod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    struct GeneratePod
    {
        ALPAKA_FN_ACC Pod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Pod{id, 0u};
        }
    };

    struct BumpPod
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod& payload) const
        {
            payload.value += 5u;
        }
    };

    struct Mod3Classifier
    {
        static constexpr std::uint32_t maxBins = 3u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % 3u);
        }
    };

    struct AddBinUpdate
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Pod& payload) const
        {
            payload.value += bin;
            return graphFlow::UpdateResult::success;
        }
    };

    template<typename TView>
    struct RecordPod
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            values[payload.id] = payload.value;
        }
    };

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }
} // namespace

TEST_CASE("particle feedback runs through the public flow", "[integration][flow][execution]")
{
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 128ull, 4097ull})
    {
        for(std::uint32_t batchChunks : {1u, 2u, 8u})
        {
            CAPTURE(particles, batchChunks);
            graphFlow::Tuning const tuning = tuningFor(32u, 8u, batchChunks);
            ParticleOutcome const outcome = runParticleFeedback(particles, tuning);
            REQUIRE(outcome.counts.size() == static_cast<std::size_t>(particles));
            for(std::uint64_t id = 0u; id < particles; ++id)
            {
                // Even ids take the feedback edge once, odd ids go straight to the sink.
                std::uint32_t const expectedVisit = id % 2u == 0u ? 2u : 1u;
                CHECK(outcome.visits[static_cast<std::size_t>(id)] == expectedVisit);
                // Exactly once: every id reaches a sink once and never a second time.
                CHECK(outcome.counts[static_cast<std::size_t>(id)] == 1u);
            }
            // The full pool is retired and no chunk leaks.
            CHECK(outcome.stats.poolChunks != 0u);
            CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
        }
    }
}

TEST_CASE("generic POD payload covers transform, partition and fan-in", "[integration][flow][execution]")
{
    for(std::uint64_t particles : {17ull, 128ull})
    {
        for(std::uint32_t batchChunks : {1u, 2u, 8u})
        {
            CAPTURE(particles, batchChunks);
            auto device = graphFlow::test::makeCpuDevice();
            auto results
                = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});

            graphFlow::Flow<Pod, Device> flow{device};
            auto source = flow.source("generate", GeneratePod{});
            auto bump = flow.transform("bump", BumpPod{});
            auto mod = flow.partition("mod3", Mod3Classifier{}, AddBinUpdate{});
            using ResultView = decltype(results.getView());
            auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
            flow.connect(source, bump);
            flow.connect(bump, mod);
            // Multiple bins target the same node.
            flow.connect(mod.out<0u>(), sink);
            flow.connect(mod.out<1u>(), sink);
            flow.connect(mod.out<2u>(), sink);
            flow.configure(tuningFor(8u, 4u, batchChunks));

            auto const stats = flow.run(source, particles);
            CHECK(stats.poolChunks != 0u);

            std::vector<std::uint32_t> const values
                = readBack<std::uint32_t>(device, results, static_cast<std::size_t>(particles));
            for(std::uint64_t id = 0u; id < particles; ++id)
                CHECK(values[static_cast<std::size_t>(id)] == 5u + static_cast<std::uint32_t>(id % 3u));
        }
    }
}

TEST_CASE("public flow keeps a configured bounded pool", "[integration][flow][execution]")
{
    auto device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t particles = 64u;
    constexpr std::uint32_t numChunks = 40u;
    auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    auto counts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        std::vector<std::uint32_t> const zero(static_cast<std::size_t>(particles), 0u);
        alpaka::onHost::memcpy(queue, visits, zero);
        alpaka::onHost::memcpy(queue, counts, zero);
    }

    graphFlow::Flow<graphFlow::Particle, Device> flow{device};
    auto source = flow.source("generate", GenerateParticle{});
    auto ingress = flow.transform("ingress", VisitIngress{});
    auto filter = flow.partition("filter", FeedbackFilter{2u});
    using VisitView = decltype(visits.getView());
    using CountView = decltype(counts.getView());
    auto sink = flow.sink("record", RecordVisit<VisitView, CountView>{visits.getView(), counts.getView()});
    flow.connect(source, ingress);
    flow.connect(ingress, filter);
    flow.connect(filter.out<0u>(), sink);
    flow.connect(filter.out<1u>(), ingress);

    graphFlow::Tuning tuning = tuningFor(16u, 8u, 2u);
    tuning.numChunks = numChunks;
    tuning.poolPayloadBudgetBytes = std::uint64_t{1u} << 24u;
    flow.configure(tuning);

    auto const stats = flow.run(source, particles);
    CHECK(stats.poolChunks == numChunks);
    CHECK(stats.peakLiveChunks <= stats.poolChunks);
    CHECK(stats.peakReservedChunks <= stats.poolChunks);

    std::vector<std::uint32_t> const readCounts
        = readBack<std::uint32_t>(device, counts, static_cast<std::size_t>(particles));
    for(std::uint32_t count : readCounts)
        CHECK(count == 1u);
}

TEST_CASE("a public flow publishes only complete outputs", "[integration][flow][execution]")
{
    // N is deliberately not a multiple of C: the last source chunk and the final transform chunks
    // are partial, but every payload must still reach the sink exactly once.
    constexpr std::uint64_t particles = 129u;
    graphFlow::Tuning const tuning = tuningFor(16u, 8u, 4u);
    ParticleOutcome const outcome = runParticleFeedback(particles, tuning);
    std::uint64_t const sunk = std::accumulate(outcome.counts.begin(), outcome.counts.end(), std::uint64_t{0u});
    CHECK(sunk == particles);
    for(std::uint64_t id = 0u; id < particles; ++id)
        CHECK(outcome.counts[static_cast<std::size_t>(id)] == 1u);
}
