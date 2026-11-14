/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#if !ALPAKA_LANG_HIP
#    error "test_flow_execution_hip.cpp requires a HIP compilation (GRAPHFLOW_ENABLE_HIP=ON)"
#endif

namespace
{
    using HipSelector = decltype(alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu));
    using HipDevice = decltype(std::declval<HipSelector&>().makeDevice(0u));

    /** Fail instead of silently falling back to the host when HIP has no visible device. */
    HipDevice makeHipDevice()
    {
        auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Hip{}, alpaka::deviceKind::amdGpu);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            throw std::runtime_error{"HIP is compiled in but no AMD GPU is visible; this test requires real hardware"};
        auto device = selector.makeDevice(0u);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Hip>);
        static_assert(std::same_as<ALPAKA_TYPEOF(device.getDeviceKind()), alpaka::deviceKind::AmdGpu>);
        return device;
    }

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
    std::vector<T> readBack(HipDevice& device, auto const& view, std::size_t size)
    {
        std::vector<T> host(size);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TQueue>
    void drain(HipDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    std::map<std::uint32_t, std::uint64_t> histogramOf(std::vector<std::uint32_t> const& values)
    {
        std::map<std::uint32_t, std::uint64_t> histogram;
        for(std::uint32_t value : values)
            ++histogram[value];
        return histogram;
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

    template<typename TView>
    struct RecordWeight
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()(auto const&, graphFlow::Particle const& particle) const
        {
            values[particle.id] = particle.weight;
        }
    };

    template<typename TVisits, typename TWeights, typename TCounts>
    struct RecordVisit
    {
        mutable TVisits visits;
        mutable TWeights weights;
        mutable TCounts counts;

        ALPAKA_FN_ACC void operator()(auto const& acc, graphFlow::Particle const& particle) const
        {
            visits[particle.id] = particle.visit;
            weights[particle.id] = particle.weight;
            alpaka::onAcc::atomicAdd(acc, &counts[particle.id], std::uint32_t{1u}, alpaka::onAcc::scope::device);
        }
    };

    struct ParticleOutcome
    {
        std::vector<std::uint32_t> visits;
        std::vector<std::uint32_t> weights;
        std::vector<std::uint32_t> counts;
        graphFlow::SchedulerStats stats;
    };

    ParticleOutcome runParticleFeedback(
        HipDevice& device,
        std::uint64_t particles,
        std::uint32_t batchChunks,
        std::uint32_t numChunks = 0u)
    {
        constexpr std::uint32_t capacity = 64u;
        constexpr std::uint32_t tileSize = 32u;
        std::size_t const size = static_cast<std::size_t>(std::max<std::uint64_t>(particles, 1u));
        auto visits = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        auto weights = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        auto counts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            std::vector<std::uint32_t> const zero(size, 0u);
            alpaka::onHost::memcpy(queue, visits, zero);
            alpaka::onHost::memcpy(queue, weights, zero);
            alpaka::onHost::memcpy(queue, counts, zero);
        }

        graphFlow::Flow<graphFlow::Particle, HipDevice> flow{device};
        auto source = flow.source("generate", GenerateParticle{});
        auto ingress = flow.transform("ingress", VisitIngress{});
        auto filter = flow.partition("filter", FeedbackFilter{2u});
        using VisitView = decltype(visits.getView());
        using WeightView = decltype(weights.getView());
        using CountView = decltype(counts.getView());
        auto sink = flow.sink(
            "record",
            RecordVisit<VisitView, WeightView, CountView>{visits.getView(), weights.getView(), counts.getView()});
        flow.connect(source, ingress);
        flow.connect(ingress, filter);
        flow.connect(filter.out<0u>(), sink);
        flow.connect(filter.out<1u>(), ingress);

        graphFlow::Tuning tuning = tuningFor(capacity, tileSize, batchChunks);
        if(particles != 0u)
        {
            // Make Q visible: admit up to Q full chunks instead of the two-chunk policy default.
            tuning.liveParticles = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(particles, static_cast<std::uint64_t>(batchChunks) * capacity));
        }
        if(numChunks != 0u)
        {
            tuning.numChunks = numChunks;
            tuning.poolPayloadBudgetBytes = std::uint64_t{1u} << 30u;
        }
        flow.configure(tuning);

        ParticleOutcome outcome;
        outcome.stats = flow.run(source, particles);
        outcome.visits = readBack<std::uint32_t>(device, visits, size);
        outcome.weights = readBack<std::uint32_t>(device, weights, size);
        outcome.counts = readBack<std::uint32_t>(device, counts, size);
        outcome.visits.resize(static_cast<std::size_t>(particles));
        outcome.weights.resize(static_cast<std::size_t>(particles));
        outcome.counts.resize(static_cast<std::size_t>(particles));
        return outcome;
    }

    void checkFeedbackOutcome(ParticleOutcome const& outcome, std::uint64_t particles)
    {
        REQUIRE(outcome.counts.size() == static_cast<std::size_t>(particles));
        for(std::uint64_t id = 0u; id < particles; ++id)
        {
            std::uint32_t const expectedVisit = id % 2u == 0u ? 2u : 1u;
            CHECK(outcome.visits[static_cast<std::size_t>(id)] == expectedVisit);
            CHECK(outcome.weights[static_cast<std::size_t>(id)] == expectedVisit);
            CHECK(outcome.counts[static_cast<std::size_t>(id)] == 1u);
        }

        std::uint64_t const expectedSum = particles + (particles + 1u) / 2u;
        std::uint64_t const sum = std::accumulate(outcome.weights.begin(), outcome.weights.end(), std::uint64_t{0u});
        CHECK(sum == expectedSum);

        auto const histogram = histogramOf(outcome.weights);
        std::uint64_t const even = (particles + 1u) / 2u;
        std::uint64_t const odd = particles / 2u;
        if(odd == 0u)
            CHECK(histogram.find(1u) == histogram.end());
        else
            CHECK(histogram.at(1u) == odd);
        if(even == 0u)
            CHECK(histogram.find(2u) == histogram.end());
        else
            CHECK(histogram.at(2u) == even);
        if(particles == 0u)
            CHECK(histogram.empty());
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
        std::uint32_t amount = 0u;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod& payload) const
        {
            payload.value += amount;
        }
    };

    struct ParityClassifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % 2u);
        }
    };

    struct AddOnOddBin
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Pod& payload) const
        {
            if(bin == 1u)
                payload.value += 1u;
            return graphFlow::UpdateResult::success;
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

    template<typename TPayload, typename TStorage>
    std::vector<TPayload> readChunk(
        HipDevice& device,
        TStorage& storage,
        graphFlow::ChunkHandle handle,
        std::uint32_t fill)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        std::vector<TPayload> host(fill);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(fill)});
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TPayload, typename TStorage>
    void writeChunk(
        HipDevice& device,
        TStorage& storage,
        graphFlow::ChunkHandle handle,
        std::vector<TPayload> const& host)
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto view = storage.payload().getSubView(
            alpaka::Vec{storage.base(handle)},
            alpaka::Vec{static_cast<std::uint64_t>(host.size())});
        alpaka::onHost::memcpy(queue, view, host);
    }
} // namespace

TEST_CASE("HIP typed flow runs the feedback matrix", "[hip][execution]")
{
    auto device = makeHipDevice();
    INFO("HIP device: " << device.getName());
    CHECK_FALSE(device.getName().empty());
    for(std::uint64_t particles : {0ull, 1ull, 17ull, 128ull, 4097ull})
    {
        for(std::uint32_t batchChunks : {1u, 2u, 8u})
        {
            CAPTURE(particles, batchChunks);
            ParticleOutcome const outcome = runParticleFeedback(device, particles, batchChunks);
            checkFeedbackOutcome(outcome, particles);
            CHECK(outcome.stats.poolChunks != 0u);
        }
    }
}

TEST_CASE("HIP typed flow handles a partial tail chunk", "[hip][execution]")
{
    auto device = makeHipDevice();
    constexpr std::uint64_t particles = 129u;
    ParticleOutcome const outcome = runParticleFeedback(device, particles, 8u);
    checkFeedbackOutcome(outcome, particles);
}

TEST_CASE("HIP typed flow keeps a bounded pool", "[hip][execution]")
{
    auto device = makeHipDevice();
    constexpr std::uint64_t particles = 128u;
    constexpr std::uint32_t numChunks = 131u;
    ParticleOutcome const outcome = runParticleFeedback(device, particles, 2u, numChunks);
    checkFeedbackOutcome(outcome, particles);
    CHECK(outcome.stats.poolChunks == numChunks);
    CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
    CHECK(outcome.stats.peakReservedChunks <= outcome.stats.poolChunks);
}

TEST_CASE("HIP typed flow source and sink handle empty and single runs", "[hip][execution]")
{
    auto device = makeHipDevice();
    for(std::uint64_t particles : {0ull, 1ull})
    {
        CAPTURE(particles);
        std::size_t const size = static_cast<std::size_t>(std::max<std::uint64_t>(particles, 1u));
        auto weights = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            std::vector<std::uint32_t> const zero(size, 0u);
            alpaka::onHost::memcpy(queue, weights, zero);
        }

        graphFlow::Flow<graphFlow::Particle, HipDevice> flow{device};
        auto source = flow.source("generate", GenerateParticle{});
        using WeightView = decltype(weights.getView());
        auto sink = flow.sink("record", RecordWeight<WeightView>{weights.getView()});
        flow.connect(source, sink);
        flow.configure(tuningFor(64u, 32u, 1u));
        auto const stats = flow.run(source, particles);
        CHECK(stats.poolChunks != 0u);

        std::vector<std::uint32_t> const read = readBack<std::uint32_t>(device, weights, size);
        for(std::size_t i = 0u; i < static_cast<std::size_t>(particles); ++i)
            CHECK(read[i] == 0u);
    }
}

TEST_CASE("HIP typed flow runs the generic POD chain", "[hip][execution]")
{
    auto device = makeHipDevice();
    for(std::uint64_t particles : {17ull, 128ull})
    {
        for(std::uint32_t batchChunks : {2u, 8u})
        {
            CAPTURE(particles, batchChunks);
            auto results
                = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::max<std::uint64_t>(particles, 1u)});

            graphFlow::Flow<Pod, HipDevice> flow{device};
            auto source = flow.source("generate", GeneratePod{});
            auto bump = flow.transform("bump", BumpPod{5u});
            auto mod = flow.partition("mod3", Mod3Classifier{}, AddBinUpdate{});
            using ResultView = decltype(results.getView());
            auto sink = flow.sink("record", RecordPod<ResultView>{results.getView()});
            flow.connect(source, bump);
            flow.connect(bump, mod);
            flow.connect(mod.out<0u>(), sink);
            flow.connect(mod.out<1u>(), sink);
            flow.connect(mod.out<2u>(), sink);
            flow.configure(tuningFor(64u, 32u, batchChunks));

            auto const stats = flow.run(source, particles);
            CHECK(stats.poolChunks != 0u);

            std::vector<std::uint32_t> const values
                = readBack<std::uint32_t>(device, results, static_cast<std::size_t>(particles));
            for(std::uint64_t id = 0u; id < particles; ++id)
                CHECK(values[static_cast<std::size_t>(id)] == 5u + static_cast<std::uint32_t>(id % 3u));
        }
    }
}

TEST_CASE("HIP typed transform stage issues one batch-wide launch", "[hip][execution]")
{
    auto device = makeHipDevice();
    using Storage = graphFlow::PayloadSlotStorage<Pod, HipDevice>;
    constexpr std::uint32_t capacity = 32u;
    Storage storage{device, 8u, capacity};

    graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
    std::vector<Pod> payloads(capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        payloads[i] = Pod{i, i};
    writeChunk(device, storage, input.handle, payloads);

    graphFlow::Config config;
    config.chunkCapacity = capacity;
    config.batchChunks = 1u;
    config.tileSize = 16u;
    config.frameCap = 2u;
    graphFlow::detail::TransformStage<HipDevice, Storage, Pod, BumpPod> stage{device, storage, config, BumpPod{3u}};
    using Runner = graphFlow::GraphRunner<HipDevice, Storage>;
    Runner::Batch const batch{{input}, 0u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    CHECK_FALSE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 1u);

    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    REQUIRE(outputs.size() == 1u);
    CHECK(stage.launches() == 1u);

    std::vector<Pod> const read = readChunk<Pod>(device, storage, outputs[0].handle, outputs[0].fill);
    REQUIRE(read.size() == capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        CHECK(read[i].value == i + 3u);

    for(graphFlow::ChunkDescriptor const& output : outputs)
        storage.release(output.handle);
    storage.release(input.handle);
}

TEST_CASE("HIP typed partition stage issues one classify and one scatter launch", "[hip][execution]")
{
    auto device = makeHipDevice();
    using Storage = graphFlow::PayloadSlotStorage<Pod, HipDevice>;
    constexpr std::uint32_t capacity = 32u;
    Storage storage{device, 12u, capacity};

    graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
    std::vector<Pod> payloads(capacity);
    for(std::uint32_t i = 0u; i < capacity; ++i)
        payloads[i] = Pod{i, 0u};
    writeChunk(device, storage, input.handle, payloads);

    graphFlow::Config config;
    config.chunkCapacity = capacity;
    config.batchChunks = 1u;
    config.tileSize = 16u;
    config.frameCap = 2u;
    graphFlow::detail::PartitionStage<HipDevice, Storage, Pod, ParityClassifier, AddOnOddBin>
        stage{device, storage, config, ParityClassifier{}, AddOnOddBin{}};
    using Runner = graphFlow::GraphRunner<HipDevice, Storage>;
    Runner::Batch const batch{{input}, 0u};
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

    stage.launch(queue, batch);
    drain(device, queue);
    std::uint32_t allocated = 0u;
    REQUIRE(stage.advance(queue, batch, allocated));
    CHECK(allocated == 2u);
    drain(device, queue);
    CHECK_FALSE(stage.advance(queue, batch, allocated));

    std::vector<graphFlow::ChunkDescriptor> outputs;
    stage.finish(batch, outputs);
    REQUIRE(outputs.size() == 2u);
    CHECK(stage.classifyLaunches() == 1u);
    CHECK(stage.scatterLaunches() == 1u);

    std::uint32_t total = 0u;
    for(graphFlow::ChunkDescriptor const& output : outputs)
    {
        std::vector<Pod> const read = readChunk<Pod>(device, storage, output.handle, output.fill);
        total += output.fill;
        for(Pod const& payload : read)
            CHECK(payload.value == (output.bin == 1u ? 1u : 0u));
    }
    CHECK(total == capacity);

    for(graphFlow::ChunkDescriptor const& output : outputs)
        storage.release(output.handle);
    storage.release(input.handle);
}

TEST_CASE("HIP typed flow feedback terminates with Q=8 and a bounded pool", "[hip][execution]")
{
    auto device = makeHipDevice();
    constexpr std::uint64_t particles = 4097u;
    constexpr std::uint32_t numChunks = 515u;
    ParticleOutcome const outcome = runParticleFeedback(device, particles, 8u, numChunks);
    checkFeedbackOutcome(outcome, particles);
    CHECK(outcome.stats.poolChunks == numChunks);
    CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
}
