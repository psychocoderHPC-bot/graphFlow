/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace
{
    /** `id % B` classifier that tags the payload with its bin so the sink can verify it. */
    template<std::uint32_t T_Bins>
    struct ModPolicy
    {
        static constexpr std::uint32_t maxBins = T_Bins;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return static_cast<std::uint32_t>(id % maxBins);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t bin,
            graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            particle.weight = bin;
        }
    };

    /** Classifier that violates its contract for every particle. */
    struct InvalidHashPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            [[maybe_unused]] std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return maxBins; // out of range on purpose
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] graphFlow::Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
        }
    };

    /** Run Generate -> Transform -> Sink with an explicit bin-table policy. */
    template<typename TDevice, typename TPolicy, typename TBinTable>
    std::vector<std::uint32_t> runPipeline(
        TDevice device,
        std::uint64_t particles,
        std::uint32_t capacity,
        TPolicy policy)
    {
        std::uint32_t const threads = graphFlow::GraphRunner<TDevice>::threadsPerBlock(device, 128u);
        std::uint64_t const reserve
            = graphFlow::outputChunkBound(capacity, capacity, TPolicy::maxBins);
        std::uint64_t const chunks = particles / capacity + 2u * reserve + 10u;
        std::uint32_t const numChunks = static_cast<std::uint32_t>(chunks);

        graphFlow::ChunkPool<TDevice> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<TDevice> generate{pool, 0u, threads};
        graphFlow::particle::TransformStage<TDevice, TPolicy, TBinTable> transform{
            device,
            pool,
            policy,
            capped,
            threads};
        graphFlow::particle::SinkStage<TDevice> sink{pool, result, threads};

        std::vector<int> route(TPolicy::maxBins, 2);
        graphFlow::GraphRunner<TDevice> runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<TDevice>(generate));
        runner.addNode(
            "transform",
            graphFlow::NodeRole::transform,
            TPolicy::maxBins,
            route,
            graphFlow::particle::hooksOf<TDevice>(transform));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<TDevice>(sink));
        runner.setSource(source, particles);
        runner.setStallTimeout(std::chrono::milliseconds{30'000});
        runner.run();

        std::vector<std::uint32_t> host(static_cast<std::size_t>(particles), 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, host, result);
        }
        return host;
    }

    template<std::uint32_t T_Bins>
    std::vector<std::uint32_t> expectedBins(std::uint64_t particles)
    {
        std::vector<std::uint32_t> expected(static_cast<std::size_t>(particles));
        for(std::uint64_t id = 0u; id < particles; ++id)
            expected[static_cast<std::size_t>(id)] = static_cast<std::uint32_t>(id % T_Bins);
        return expected;
    }
} // namespace

TEST_CASE("hash bin table is a compile-time multiple of the tile", "[bintable][hash]")
{
    CHECK(graphFlow::DenseBinTable::isHash == false);
    CHECK(graphFlow::HashBinTable<64u>::entries == 128u);
    CHECK(graphFlow::HashBinTable<128u>::entries == 256u);
    CHECK(graphFlow::HashBinTable<128u>::mask == 255u);
    CHECK(graphFlow::HashBinTable<128u, 4u>::entries == 512u);
    CHECK(graphFlow::HashBinTable<100u>::requestedEntries == 200u);
    CHECK(graphFlow::HashBinTable<100u>::entries == 256u);
}

TEST_CASE("hash and dense pipelines agree at overlapping bin counts", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();

    for(std::uint32_t const bins : {64u, 1024u})
    {
        CAPTURE(bins);
        constexpr std::uint64_t particles = 2000u;
        if(bins == 64u)
        {
            std::vector<std::uint32_t> const dense
                = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<64u>, graphFlow::DenseBinTable>(
                    device,
                    particles,
                    17u,
                    ModPolicy<64u>{});
            std::vector<std::uint32_t> const hash
                = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<64u>, graphFlow::HashBinTable<128u>>(
                    device,
                    particles,
                    17u,
                    ModPolicy<64u>{});
            CHECK(hash == dense);
            CHECK(hash == expectedBins<64u>(particles));
        }
        else
        {
            std::vector<std::uint32_t> const dense
                = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<1024u>, graphFlow::DenseBinTable>(
                    device,
                    particles,
                    31u,
                    ModPolicy<1024u>{});
            std::vector<std::uint32_t> const hash
                = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<1024u>, graphFlow::HashBinTable<128u>>(
                    device,
                    particles,
                    31u,
                    ModPolicy<1024u>{});
            CHECK(hash == dense);
            CHECK(hash == expectedBins<1024u>(particles));
        }
    }
}

TEST_CASE("hash pipeline handles sparse 4096 and 65536 bin distributions", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    constexpr std::uint64_t particles = 3000u;
    constexpr std::uint32_t capacity = 64u;

    std::vector<std::uint32_t> const sparse4096
        = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<4096u>, graphFlow::HashBinTable<128u>>(
            device,
            particles,
            capacity,
            ModPolicy<4096u>{});
    CHECK(sparse4096 == expectedBins<4096u>(particles));

    std::vector<std::uint32_t> const sparse65536
        = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<65536u>, graphFlow::HashBinTable<128u>>(
            device,
            particles,
            capacity,
            ModPolicy<65536u>{});
    CHECK(sparse65536 == expectedBins<65536u>(particles));
}

TEST_CASE("hash pipeline crosses chunk boundaries", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    constexpr std::uint64_t particles = 50u;
    constexpr std::uint32_t capacity = 3u;

    std::vector<std::uint32_t> const hash
        = runPipeline<Device, ModPolicy<8u>, graphFlow::HashBinTable<128u>>(
            device,
            particles,
            capacity,
            ModPolicy<8u>{});
    CHECK(hash == expectedBins<8u>(particles));

    // The dense reference must produce the same id-indexed result.
    std::vector<std::uint32_t> const dense
        = runPipeline<Device, ModPolicy<8u>, graphFlow::DenseBinTable>(
            device,
            particles,
            capacity,
            ModPolicy<8u>{});
    CHECK(hash == dense);
}

TEST_CASE("hash table exhaustion aborts the batch", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    std::uint32_t const threads = graphFlow::GraphRunner<Device>::threadsPerBlock(device, 128u);

    // Eight slots, sixteen distinct bins in the only source chunk: the table is full.
    constexpr std::uint32_t numChunks = 64u;
    constexpr std::uint32_t capacity = 16u;
    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, ModPolicy<16u>, graphFlow::HashBinTable<4u>> transform{
        device,
        pool,
        ModPolicy<16u>{},
        capped,
        threads};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

    std::vector<int> route(ModPolicy<16u>::maxBins, 2);
    graphFlow::GraphRunner<Device> runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<Device>(generate));
    runner.addNode(
        "transform",
        graphFlow::NodeRole::transform,
        ModPolicy<16u>::maxBins,
        route,
        graphFlow::particle::hooksOf<Device>(transform));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10'000});

    CHECK_THROWS_AS(runner.run(), graphFlow::BatchError);
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLive() <= pool.numChunks());

    // No output was published: the sentinel survives every id.
    std::vector<std::uint32_t> host(16u, 0u);
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, result);
    }
    for(std::uint32_t const value : host)
        CHECK(value == 0xDEADBEEFu);

    // Control: the same batch succeeds once the table is large enough for all sixteen bins, so
    // the error above is capacity-driven and not an unrelated batch failure.
    std::vector<std::uint32_t> const control
        = runPipeline<Device, ModPolicy<16u>, graphFlow::HashBinTable<16u>>(
            device,
            16u,
            capacity,
            ModPolicy<16u>{});
    CHECK(control == expectedBins<16u>(16u));
}

TEST_CASE("invalid classifier results abort the hash batch", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    std::uint32_t const threads = graphFlow::GraphRunner<Device>::threadsPerBlock(device, 128u);

    graphFlow::ChunkPool pool{device, 32u, 8u};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEADBEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{pool, 0u, threads};
    graphFlow::particle::TransformStage<Device, InvalidHashPolicy, graphFlow::HashBinTable<128u>> transform{
        device,
        pool,
        InvalidHashPolicy{},
        capped,
        threads};
    graphFlow::particle::SinkStage<Device> sink{pool, result, threads};

    graphFlow::GraphRunner<Device> runner{device, pool};
    auto const source = runner.addNode(
        "generate",
        graphFlow::NodeRole::source,
        1u,
        {1},
        graphFlow::particle::hooksOf<Device>(generate));
    runner.addNode(
        "invalid",
        graphFlow::NodeRole::transform,
        InvalidHashPolicy::maxBins,
        {2, 2},
        graphFlow::particle::hooksOf<Device>(transform));
    runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<Device>(sink));
    runner.setSource(source, 16u);
    runner.setStallTimeout(std::chrono::milliseconds{10'000});

    CHECK_THROWS_AS(runner.run(), graphFlow::BatchError);
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLive() <= pool.numChunks());
}
