/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ParticleFeedback.hpp"
#include "ParticleStages.hpp"
#include "ScalarModel.hpp"
#include "TestDevice.hpp"
#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/graph/ClassifyKernel.hpp"
#include "graphFlow/graph/TileDescriptor.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace
{
    template<typename TDevice>
    void checkAgainstScalar(TDevice& device, graphFlow::particle::Parameters const& parameters)
    {
        auto const result = graphFlow::particle::runParticleFeedback(device, parameters);
        auto const reference
            = graphFlow::test::runScalarModel(parameters.seed, parameters.numParticles, parameters.maxVisits);

        CHECK(result.histogram == graphFlow::particle::histogramOf(reference.weights));
        CHECK(result.weights == reference.weights);
        CHECK(result.sunkParticles == parameters.numParticles);
        CHECK(result.cappedParticles == reference.cappedParticles);
    }

    /** N matrix, tail values included. */
    template<typename TDevice>
    void runParticleCountMatrix(TDevice& device)
    {
        for(std::uint64_t const particles : {0ull, 1ull, 17ull, 127ull, 128ull, 4097ull})
        {
            graphFlow::particle::Parameters parameters{};
            parameters.numParticles = particles;
            parameters.chunkSize = 128u;
            parameters.tileSize = 128u;
            parameters.batchChunks = 2u;
            parameters.frameCap = 8u;
            parameters.seed = 42u;
            parameters.maxVisits = 64u;

            CAPTURE(particles);
            checkAgainstScalar(device, parameters);
        }
    }

    /** Capacity and tile matrix through the dense path. */
    template<typename TDevice>
    void runCapacityTileMatrix(TDevice& device)
    {
        for(std::uint32_t const capacity : {1u, 31u, 127u, 128u})
        {
            for(std::uint32_t const tileSize : {32u, 128u})
            {
                graphFlow::particle::Parameters parameters{};
                parameters.numParticles = 4097u;
                parameters.chunkSize = capacity;
                parameters.tileSize = tileSize;
                parameters.batchChunks = 2u;
                parameters.frameCap = 8u;
                parameters.seed = 42u;
                parameters.maxVisits = 64u;

                CAPTURE(capacity, tileSize);
                checkAgainstScalar(device, parameters);
            }
        }
    }

    /** frameCap and Q matrix: every run keeps histogram equality with the scalar path. */
    template<typename TDevice>
    void runFrameCapBatchMatrix(TDevice& device)
    {
        for(std::uint32_t const frameCap : {1u, 8u, std::numeric_limits<std::uint32_t>::max()})
        {
            for(std::uint32_t const batchChunks : {1u, 2u, 8u})
            {
                graphFlow::particle::Parameters parameters{};
                parameters.numParticles = 4097u;
                parameters.chunkSize = 128u;
                parameters.tileSize = 32u;
                parameters.batchChunks = batchChunks;
                parameters.frameCap = frameCap;
                parameters.seed = 42u;
                parameters.maxVisits = 64u;

                CAPTURE(frameCap, batchChunks);
                checkAgainstScalar(device, parameters);
            }
        }
    }

    struct TwoBinPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            [[maybe_unused]] std::uint32_t visit) const
        {
            return static_cast<std::uint32_t>(id % 2u);
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            std::uint32_t,
            graphFlow::Particle&,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
        }
    };
} // namespace

TEST_CASE("FrameSpec handles the N, C, T, frameCap and Q matrices", "[integration][framespec]")
{
    auto device = graphFlow::test::makeCpuDevice();
    runParticleCountMatrix(device);
    runCapacityTileMatrix(device);
    runFrameCapBatchMatrix(device);
}

TEST_CASE("one block processes many tile descriptors", "[integration][framespec]")
{
    auto device = graphFlow::test::makeCpuDevice();
    using Device = ALPAKA_TYPEOF(device);
    using Runner = graphFlow::GraphRunner<Device>;

    constexpr std::uint32_t capacity = 8u;
    constexpr std::uint32_t chunks = 4u;
    constexpr std::uint32_t tileSize = 8u;
    graphFlow::ChunkPool<Device> pool{device, chunks, capacity};

    auto initQueue = device.makeQueue(alpaka::queueKind::nonBlocking);
    graphFlow::particle::GenerateStage<Device> generate{device, pool, 0u, tileSize, 1u};
    for(std::uint32_t index = 0u; index < chunks; ++index)
    {
        graphFlow::ChunkDescriptor const chunk{pool.acquire(), capacity, 0u};
        Runner::Batch const batch{{chunk}, static_cast<std::uint64_t>(index) * capacity};
        generate.launch(initQueue, batch);
    }
    alpaka::onHost::wait(initQueue);

    auto counts = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{3u}});
    auto tiles = alpaka::onHost::alloc<graphFlow::TileDescriptor>(device, alpaka::Vec{std::uint64_t{chunks}});
    std::vector<graphFlow::TileDescriptor> hostTiles;
    for(std::uint32_t index = 0u; index < chunks; ++index)
        hostTiles.push_back(graphFlow::TileDescriptor{index, 0u, capacity});

    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    alpaka::onHost::memset(queue, counts, std::uint8_t{0});
    alpaka::onHost::memcpy(queue, tiles, hostTiles);
    queue.enqueue(
        alpaka::onHost::FrameSpec{std::uint32_t{1u}, std::uint32_t{1u}},
        alpaka::KernelBundle{
            graphFlow::ClassifyKernel<TwoBinPolicy>{TwoBinPolicy{}},
            pool.ids(),
            pool.properties(),
            pool.weights(),
            pool.visits(),
            pool.binIds(),
            counts,
            tiles,
            capacity});

    std::vector<std::uint32_t> hostCounts(3u, 0u);
    alpaka::onHost::memcpy(queue, hostCounts, counts);
    CHECK(hostCounts[0] == chunks * capacity / 2u);
    CHECK(hostCounts[1] == chunks * capacity / 2u);
    CHECK(hostCounts[2] == 0u);
}

#if ALPAKA_LANG_CUDA
TEST_CASE("FrameSpec particle matrix matches the scalar oracle on CUDA", "[cuda]")
{
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::api::Cuda{}, alpaka::deviceKind::nvidiaGpu);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        return;
    auto device = selector.makeDevice(0u);
    runParticleCountMatrix(device);
    runCapacityTileMatrix(device);
    runFrameCapBatchMatrix(device);
}
#endif
