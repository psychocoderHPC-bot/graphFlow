/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "TestDevice.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    using Device = decltype(graphFlow::test::makeCpuDevice());

    /** Payload of the parity graph. `id` is the batch-global particle index. */
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

    /** Payload-dependent transform, so a read from a wrong base produces a wrong value. */
    struct StampPod
    {
        std::uint32_t salt = 0u;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Pod& payload) const
        {
            payload.value = static_cast<std::uint32_t>(payload.id % 97u) + salt;
        }
    };

    template<std::uint32_t TBins>
    struct BinOf
    {
        static constexpr std::uint32_t maxBins = TBins;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Pod const& payload) const
        {
            return static_cast<std::uint32_t>(payload.id % TBins);
        }
    };

    struct MarkBin
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Pod& payload) const
        {
            payload.value += bin * 1000u;
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

    struct Run
    {
        std::vector<std::uint32_t> values;
        graphFlow::SchedulerStats stats;
    };

    graphFlow::Tuning tuningFor(std::uint32_t capacity, std::uint32_t batchChunks)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = 4u;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        return tuning;
    }

    /** Run source -> transform -> partition -> sink through the public Flow API.
     *
     * `TBins` is the partition width as well as the number of sink edges, so every configured
     * bin reaches the sink and the graph writes each id exactly once.
     */
    template<std::uint32_t TBins, std::size_t... TPorts>
    void connectBins(auto& flow, auto& partition, auto& sink, std::index_sequence<TPorts...>)
    {
        (flow.connect(partition.template out<TPorts>(), sink), ...);
    }

    template<std::uint32_t TBins>
    Run runGraph(std::uint64_t particles, graphFlow::Tuning const& tuning)
    {
        auto device = graphFlow::test::makeCpuDevice();
        std::uint64_t const resultSize = particles == 0u ? 1u : particles;
        auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{resultSize});

        graphFlow::Flow<Pod, Device> flow{device};
        auto source = flow.source("generate", GeneratePod{});
        auto stamp = flow.transform("stamp", StampPod{3u});
        auto bins = flow.partition("bins", BinOf<TBins>{}, MarkBin{});
        auto sink = flow.sink("record", RecordPod<decltype(results.getView())>{results.getView()});
        flow.connect(source, stamp);
        flow.connect(stamp, bins);
        connectBins<TBins>(flow, bins, sink, std::make_index_sequence<TBins>{});
        flow.configure(tuning);

        Run run;
        run.stats = flow.run(source, particles);

        std::vector<std::uint32_t> host(static_cast<std::size_t>(resultSize));
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, results);
        host.resize(static_cast<std::size_t>(particles));
        run.values = std::move(host);
        return run;
    }

    template<std::uint32_t TBins>
    void checkReference(std::vector<std::uint32_t> const& values)
    {
        for(std::size_t id = 0u; id < values.size(); ++id)
        {
            std::uint32_t const expected = static_cast<std::uint32_t>(id % 97u) + 3u
                + static_cast<std::uint32_t>(id % TBins) * 1000u;
            CAPTURE(id);
            CHECK(values[id] == expected);
        }
    }

    /** Compare the physics of the fixed and the slot-arena layout of one configuration. */
    template<std::uint32_t TBins>
    Run checkArenaParity(std::uint64_t particles, graphFlow::Tuning tuning, char const* schedule)
    {
        CAPTURE(schedule, particles);
        Run const fixed = runGraph<TBins>(particles, tuning);
        tuning.poolMode = graphFlow::PoolMode::slotArena;
        Run const arena = runGraph<TBins>(particles, tuning);

        REQUIRE(fixed.values.size() == particles);
        REQUIRE(arena.values.size() == particles);
        CHECK(arena.values == fixed.values);
        checkReference<TBins>(fixed.values);
        checkReference<TBins>(arena.values);
        // The chunk-index count is P + K*R in both layouts; only the slot count differs.
        CHECK(arena.stats.poolChunks == fixed.stats.poolChunks);
        CHECK(arena.stats.sourceSunk == particles);
        CHECK(arena.stats.sourceSunk == fixed.stats.sourceSunk);
        return arena;
    }
} // namespace

TEST_CASE(
    "arena matches fixed physics with a batch-wide transform",
    "[integration][arena][transform]")
{
    checkArenaParity<5u>(2048u, tuningFor(16u, 2u), "batch-wide");
}

TEST_CASE(
    "arena matches fixed physics with the per-chunk transform schedule",
    "[integration][arena][transform]")
{
    // A small admission capacity keeps the chunk indices in rotation, so the arena hands out
    // reused ranges whose base is not the chunk index times the capacity.
    graphFlow::Tuning tuning = tuningFor(8u, 1u);
    tuning.liveParticles = 4u;
    tuning.chunkKernelSchedule = graphFlow::ChunkKernelSchedule::perChunk;
    checkArenaParity<5u>(4096u, tuning, "perChunk");
}

TEST_CASE(
    "arena matches fixed physics on the hash path with a fragmented stream",
    "[integration][arena][hash]")
{
    // An explicit hash request selects the hash kernels for any width; C=8 lets every source
    // chunk split into up to one chunk per bin, so the live chunk count grows downstream while
    // the arena is in use.
    constexpr std::uint32_t bins = 8u;
    constexpr std::uint64_t particles = 2048u;
    graphFlow::Tuning tuning = tuningFor(8u, 2u);
    tuning.tileSize = 8u;
    tuning.binPath = graphFlow::BinPath::hash;
    Run const arena = checkArenaParity<bins>(particles, tuning, "hash");

    // Downstream fan-out: at most 2048/8 = 256 source chunks, and every source chunk splits
    // into up to eight output chunks. Acquisitions count the chunks the run actually took.
    CHECK(arena.stats.chunkRegistryAcquisitions > particles / tuning.chunkCapacity.value());
    CHECK(arena.stats.peakLiveChunks > 1u);
}
