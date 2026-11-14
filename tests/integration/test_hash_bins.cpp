/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

    /** Run Generate -> Transform -> Sink with an explicit bin-table policy.
     *
     * `numChunks == 0` uses a generous test pool. A caller-supplied pool lets the tests pin the
     * production bound `P + K*R(C, C, maxBins)` and check that a hash partition never acquires
     * more output credits than the scheduler reserved for it.
     */
    template<typename TDevice, typename TPolicy, typename TBinTable>
    std::vector<std::uint32_t> runPipeline(
        TDevice device,
        std::uint64_t particles,
        std::uint32_t capacity,
        [[maybe_unused]] TPolicy policy,
        std::uint32_t batchChunks = 1u,
        std::uint32_t liveParticles = 0u,
        std::uint32_t numChunks = 0u,
        graphFlow::SchedulerStats* statsOut = nullptr)
    {
        std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, TPolicy::maxBins);
        std::uint64_t const chunks = numChunks != 0u ? numChunks : particles / capacity + 2u * reserve + 10u;

        graphFlow::ChunkPool<TDevice> pool{device, static_cast<std::uint32_t>(chunks), capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<TDevice> generate{device, pool, 0u, 128u};
        // The transform takes the logical tile width T and resolves the block extent itself.
        graphFlow::particle::TransformStage<TDevice, TPolicy, TBinTable>
            transform{device, pool, policy, capped, 128u, batchChunks};
        graphFlow::particle::SinkStage<TDevice> sink{device, pool, result, 128u};

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
        runner.setSource(source, particles, liveParticles);
        runner.setBatchChunks(batchChunks);
        runner.setStallTimeout(std::chrono::milliseconds{30000});
        graphFlow::SchedulerStats const stats = runner.run();
        if(statsOut != nullptr)
            *statsOut = stats;

        std::vector<std::uint32_t> host(static_cast<std::size_t>(particles), 0u);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, host, result);
        }
        return host;
    }

    /** Split one source chunk into `TSplitPolicy::maxBins` chunks, then feed a hash partition.
     *
     * The split makes the join's full Q-ready prefix visible in one completion, so the test
     * exercises a Q-chunk hash batch rather than a sequence of single-chunk batches.
     */
    template<typename TDevice, typename TSplitPolicy, typename TJoinPolicy>
    std::vector<std::uint32_t> runSplitHashPipeline(
        TDevice device,
        std::uint64_t particles,
        std::uint32_t capacity,
        TSplitPolicy splitPolicy,
        TJoinPolicy joinPolicy,
        std::uint32_t batchChunks,
        std::uint32_t liveParticles,
        std::uint32_t numChunks,
        graphFlow::SchedulerStats* statsOut,
        std::uint32_t* joinMaxBatchOut)
    {
        graphFlow::ChunkPool<TDevice> pool{device, numChunks, capacity};
        auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{particles});
        auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
        {
            auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
            alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
        }

        graphFlow::particle::GenerateStage<TDevice> generate{device, pool, 0u, 128u};
        graphFlow::particle::TransformStage<TDevice, TSplitPolicy, graphFlow::DenseBinTable>
            split{device, pool, splitPolicy, capped, 128u, batchChunks};
        graphFlow::particle::TransformStage<TDevice, TJoinPolicy, graphFlow::HashBinTable<128u>>
            join{device, pool, joinPolicy, capped, 128u, batchChunks};
        graphFlow::particle::SinkStage<TDevice> sink{device, pool, result, 128u};

        // The stage's fixed bound is one full Q-chunk batch with the partition maxBins. The hash
        // table entry count and the tile width do not enter it.
        CHECK(
            split.reserveBound()
            == graphFlow::outputChunkBound(std::uint64_t{capacity} * batchChunks, capacity, TSplitPolicy::maxBins));
        CHECK(
            join.reserveBound()
            == graphFlow::outputChunkBound(std::uint64_t{capacity} * batchChunks, capacity, TJoinPolicy::maxBins));

        // Node ids: source=0, split=1, join=2, sink=3.
        std::vector<int> splitRoute(TSplitPolicy::maxBins, 2);
        std::vector<int> joinRoute(TJoinPolicy::maxBins, 3);
        graphFlow::GraphRunner<TDevice> runner{device, pool};
        auto const source = runner.addNode(
            "generate",
            graphFlow::NodeRole::source,
            1u,
            {1},
            graphFlow::particle::hooksOf<TDevice>(generate));
        runner.addNode(
            "split",
            graphFlow::NodeRole::transform,
            TSplitPolicy::maxBins,
            splitRoute,
            graphFlow::particle::hooksOf<TDevice>(split));
        runner.addNode(
            "join",
            graphFlow::NodeRole::transform,
            TJoinPolicy::maxBins,
            joinRoute,
            graphFlow::particle::hooksOf<TDevice>(join));
        runner.addNode("sink", graphFlow::NodeRole::sink, 1u, {}, graphFlow::particle::hooksOf<TDevice>(sink));
        runner.setSource(source, particles, liveParticles);
        runner.setBatchChunks(batchChunks);
        runner.setStallTimeout(std::chrono::milliseconds{30000});
        graphFlow::SchedulerStats const stats = runner.run();
        if(statsOut != nullptr)
            *statsOut = stats;
        if(joinMaxBatchOut != nullptr)
            *joinMaxBatchOut = join.maxBatchChunks();

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

TEST_CASE("dense and hash pipelines tile a chunk larger than T", "[integration][hash]")
{
    // Eight blocks per chunk: seven full 128-particle tiles plus one full or partial tail tile.
    struct Case
    {
        std::uint64_t particles;
        std::uint32_t capacity;
    };

    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    for(Case const& testCase : {Case{1024u, 1024u}, Case{1000u, 1000u}})
    {
        CAPTURE(testCase.particles, testCase.capacity);
        std::vector<std::uint32_t> const dense = runPipeline<Device, ModPolicy<64u>, graphFlow::DenseBinTable>(
            device,
            testCase.particles,
            testCase.capacity,
            ModPolicy<64u>{});
        std::vector<std::uint32_t> const hash = runPipeline<Device, ModPolicy<64u>, graphFlow::HashBinTable<128u>>(
            device,
            testCase.particles,
            testCase.capacity,
            ModPolicy<64u>{});
        std::vector<std::uint32_t> const expected = expectedBins<64u>(testCase.particles);
        CHECK(hash == dense);
        CHECK(hash == expected);

        std::vector<std::uint32_t> sortedHash = hash;
        std::vector<std::uint32_t> sortedExpected = expected;
        std::sort(sortedHash.begin(), sortedHash.end());
        std::sort(sortedExpected.begin(), sortedExpected.end());
        CHECK(sortedHash == sortedExpected);
        CHECK(graphFlow::particle::histogramOf(hash) == graphFlow::particle::histogramOf(expected));
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

    std::vector<std::uint32_t> const hash = runPipeline<Device, ModPolicy<8u>, graphFlow::HashBinTable<128u>>(
        device,
        particles,
        capacity,
        ModPolicy<8u>{});
    CHECK(hash == expectedBins<8u>(particles));

    // The dense reference must produce the same id-indexed result.
    std::vector<std::uint32_t> const dense
        = runPipeline<Device, ModPolicy<8u>, graphFlow::DenseBinTable>(device, particles, capacity, ModPolicy<8u>{});
    CHECK(hash == dense);
}

TEST_CASE("hash table exhaustion aborts the batch", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);

    // Eight slots, sixteen distinct bins in the only source chunk: the table is full.
    constexpr std::uint32_t numChunks = 64u;
    constexpr std::uint32_t capacity = 16u;
    graphFlow::ChunkPool pool{device, numChunks, capacity};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, ModPolicy<16u>, graphFlow::HashBinTable<4u>>
        transform{device, pool, ModPolicy<16u>{}, capped, 128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

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
    runner.setStallTimeout(std::chrono::milliseconds{10000});

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
        CHECK(value == 0xDEAD'BEEFu);

    // Control: the same batch succeeds once the table is large enough for all sixteen bins, so
    // the error above is capacity-driven and not an unrelated batch failure.
    std::vector<std::uint32_t> const control
        = runPipeline<Device, ModPolicy<16u>, graphFlow::HashBinTable<16u>>(device, 16u, capacity, ModPolicy<16u>{});
    CHECK(control == expectedBins<16u>(16u));
}

TEST_CASE("invalid classifier results abort the hash batch", "[integration][hash]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);

    graphFlow::ChunkPool pool{device, 32u, 8u};
    auto result = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{16u}});
    auto capped = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}});
    {
        auto initQueue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::fill(initQueue, result, std::uint32_t{0xDEAD'BEEFu});
        alpaka::onHost::memset(initQueue, capped, std::uint8_t{0});
    }

    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, 128u};
    graphFlow::particle::TransformStage<Device, InvalidHashPolicy, graphFlow::HashBinTable<128u>>
        transform{device, pool, InvalidHashPolicy{}, capped, 128u};
    graphFlow::particle::SinkStage<Device> sink{device, pool, result, 128u};

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
    runner.setStallTimeout(std::chrono::milliseconds{10000});

    CHECK_THROWS_AS(runner.run(), graphFlow::BatchError);
    CHECK(pool.freeCount() == pool.numChunks());
    CHECK(pool.peakLive() <= pool.numChunks());
}

TEST_CASE("hash reserve uses maxBins above the dense limit for Q=1/2/8", "[integration][hash][reserve]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);

    // C=512 and one source chunk: the split emits eight ready 64-particle chunks, so the join's
    // Q=8 batch carries 512 particles. Every id is a distinct bin below 65536, so that batch
    // needs exactly 512 output chunks. B=65536 is far above Config::denseBinLimit, the
    // 256-entry shared hash table, the 128-wide tile, and Q. Using any of those as the credit
    // under-reserves the batch; the correct maxBins bound is the full batch particle count.
    constexpr std::uint32_t capacity = 512u;
    constexpr std::uint32_t splitBins = 8u;
    constexpr std::uint32_t joinBins = 65536u;
    constexpr std::uint64_t particles = 512u;
    constexpr std::uint32_t liveParticles = 512u;
    std::uint64_t const splitReserve = graphFlow::outputChunkBound(capacity, capacity, splitBins);
    std::uint64_t const joinReserve = graphFlow::outputChunkBound(capacity, capacity, joinBins);
    std::uint32_t const poolChunks = static_cast<std::uint32_t>(liveParticles + splitReserve + joinReserve);

    for(std::uint32_t const batchChunks : {1u, 2u, 8u})
    {
        CAPTURE(batchChunks);
        graphFlow::SchedulerStats stats{};
        std::uint32_t joinMaxBatch = 0u;
        std::vector<std::uint32_t> const hash
            = runSplitHashPipeline<Device, ModPolicy<splitBins>, ModPolicy<joinBins>>(
                device,
                particles,
                capacity,
                ModPolicy<splitBins>{},
                ModPolicy<joinBins>{},
                batchChunks,
                liveParticles,
                poolChunks,
                &stats,
                &joinMaxBatch);

        CHECK(hash == expectedBins<joinBins>(particles));
        // The join really saw a full Q-chunk prefix, so the tight-credit path is exercised.
        CHECK(joinMaxBatch == batchChunks);
        CHECK(stats.poolChunks == poolChunks);
        CHECK(stats.peakReservedChunks <= poolChunks);
        CHECK(stats.peakReservedChunks >= (particles / splitBins) * batchChunks);
    }
}

TEST_CASE("hash partial batches reserve the accepted prefix only", "[integration][hash][reserve]")
{
    auto device = graphFlow::test::makeCpuDevice();
    constexpr std::uint32_t capacity = 64u;
    constexpr std::uint32_t maxBins = 4096u;
    constexpr std::uint64_t particles = 1000u; // 15 full chunks plus a 40-particle tail
    constexpr std::uint32_t liveParticles = 1000u;
    std::uint64_t const reserve = graphFlow::outputChunkBound(capacity, capacity, maxBins);
    std::uint32_t const poolChunks = static_cast<std::uint32_t>(liveParticles + reserve);

    graphFlow::SchedulerStats stats{};
    std::vector<std::uint32_t> const hash
        = runPipeline<ALPAKA_TYPEOF(device), ModPolicy<maxBins>, graphFlow::HashBinTable<128u>>(
            device,
            particles,
            capacity,
            ModPolicy<maxBins>{},
            8u,
            liveParticles,
            poolChunks,
            &stats);

    CHECK(hash == expectedBins<maxBins>(particles));
    CHECK(stats.poolChunks == poolChunks);
    CHECK(stats.peakReservedChunks <= poolChunks);
}
