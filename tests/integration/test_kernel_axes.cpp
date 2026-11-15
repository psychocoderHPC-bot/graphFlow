/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "KernelAxes.hpp"
#include "ParticleTypes.hpp"
#include "TestDevice.hpp"
#include "graphFlow/graph/ScatterKernel.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{
    using graphFlow::particle::BinPath;
    using graphFlow::particle::Distribution;
    using graphFlow::particle::HistogramRow;
    using graphFlow::particle::KernelOptions;
    using graphFlow::particle::Parameters;
    using graphFlow::particle::ScatterChoice;

    /** Host-side occupancy of the id-keyed distribution mapping. */
    std::vector<std::uint64_t> expectedCounts(Distribution distribution, std::uint64_t particles, std::uint32_t bins)
    {
        std::vector<std::uint64_t> counts(bins, 0u);
        for(std::uint64_t id = 0u; id < particles; ++id)
        {
            std::uint32_t bin = 0u;
            switch(distribution)
            {
            case Distribution::oneHot:
                bin = 0u;
                break;
            case Distribution::skewed:
                bin = id % 8u == 0u ? static_cast<std::uint32_t>(id % bins) : 0u;
                break;
            case Distribution::balanced:
            case Distribution::sparse:
            default:
                bin = static_cast<std::uint32_t>(id % bins);
                break;
            }
            ++counts[bin];
        }
        return counts;
    }

    std::vector<HistogramRow> expectedHistogram(Distribution distribution, std::uint64_t particles, std::uint32_t bins)
    {
        std::vector<HistogramRow> rows;
        std::vector<std::uint64_t> const counts = expectedCounts(distribution, particles, bins);
        for(std::uint32_t bin = 0u; bin < bins; ++bin)
        {
            if(counts[bin] != 0u)
                rows.push_back(HistogramRow{bin, counts[bin]});
        }
        return rows;
    }

    /** Run one kernel-workload case and assert the exact occupancy and conservation contract. */
    void checkCase(Distribution distribution, BinPath path, ScatterChoice scatter, std::uint32_t bins)
    {
        auto device = graphFlow::test::makeCpuDevice();
        constexpr std::uint64_t particles = 2000u;
        constexpr std::uint32_t capacity = 17u;

        Parameters parameters{};
        parameters.numParticles = particles;
        parameters.chunkSize = capacity;
        parameters.liveParticles = static_cast<std::uint32_t>(particles);
        parameters.tileSize = 128u;
        parameters.batchChunks = 1u;

        KernelOptions options{};
        options.distribution = distribution;
        options.binPath = path;
        options.scatter = scatter;
        options.bins = bins;

        CAPTURE(static_cast<int>(distribution), static_cast<int>(path), static_cast<int>(scatter), bins);

        graphFlow::particle::KernelAxisResult const result
            = graphFlow::particle::runKernelAxes(device, parameters, options);

        CHECK(result.histogram == expectedHistogram(distribution, particles, bins));
        REQUIRE(result.weights.size() == particles);
        std::uint64_t sum = 0u;
        for(HistogramRow const& row : result.histogram)
            sum += row.count;
        CHECK(sum == particles);
        // The sink never leaves the missing-weight sentinel behind for a particle that was sunk.
        CHECK(
            std::none_of(
                result.weights.begin(),
                result.weights.end(),
                [](std::uint32_t weight) { return weight == graphFlow::particle::missingWeight; }));
    }
} // namespace

TEST_CASE("kernel workload defaults to the shared dense path", "[integration][kernel-axes]")
{
    using SharedPolicy = graphFlow::particle::KernelDistributionPolicy<64u, graphFlow::ScatterMode::SharedStaging>;
    using DirectPolicy = graphFlow::particle::KernelDistributionPolicy<64u, graphFlow::ScatterMode::Direct>;

    CHECK(graphFlow::detail::ScatterModeOf<SharedPolicy>::value == graphFlow::ScatterMode::SharedStaging);
    CHECK(graphFlow::detail::ScatterModeOf<DirectPolicy>::value == graphFlow::ScatterMode::Direct);

    KernelOptions const defaults{};
    CHECK(defaults.scatter == ScatterChoice::shared);
    CHECK(defaults.binPath == BinPath::dense);
    CHECK(defaults.bins == 64u);
    CHECK(defaults.distribution == Distribution::balanced);
}

TEST_CASE("kernel workload reproduces every distribution exactly", "[integration][kernel-axes]")
{
    checkCase(Distribution::balanced, BinPath::dense, ScatterChoice::shared, 64u);
    checkCase(Distribution::oneHot, BinPath::dense, ScatterChoice::shared, 64u);
    checkCase(Distribution::skewed, BinPath::dense, ScatterChoice::shared, 64u);
    checkCase(Distribution::balanced, BinPath::dense, ScatterChoice::shared, 1024u);
}

TEST_CASE("kernel workload direct and shared scatter agree", "[integration][kernel-axes]")
{
    checkCase(Distribution::balanced, BinPath::dense, ScatterChoice::direct, 64u);
    checkCase(Distribution::oneHot, BinPath::dense, ScatterChoice::direct, 64u);
    checkCase(Distribution::skewed, BinPath::dense, ScatterChoice::direct, 64u);
    checkCase(Distribution::balanced, BinPath::dense, ScatterChoice::direct, 1024u);
}

TEST_CASE("kernel workload dense and hash paths agree at overlapping bins", "[integration][kernel-axes]")
{
    checkCase(Distribution::balanced, BinPath::hash, ScatterChoice::shared, 64u);
    checkCase(Distribution::balanced, BinPath::hash, ScatterChoice::shared, 1024u);
}

TEST_CASE("kernel workload hash path handles sparse wide bins", "[integration][kernel-axes]")
{
    checkCase(Distribution::sparse, BinPath::hash, ScatterChoice::shared, 4096u);
    checkCase(Distribution::sparse, BinPath::hash, ScatterChoice::shared, 65536u);
}

TEST_CASE("kernel workload rejects unsupported public-Flow axis combinations", "[integration][kernel-axes]")
{
    auto device = graphFlow::test::makeCpuDevice();

    Parameters parameters{};
    parameters.numParticles = 64u;
    parameters.chunkSize = 16u;
    parameters.liveParticles = 64u;
    parameters.tileSize = 128u;
    parameters.batchChunks = 1u;

    KernelOptions options{};

    options.bins = 4096u;
    options.binPath = BinPath::dense;
    CHECK_THROWS_WITH(
        graphFlow::particle::runKernelAxes(device, parameters, options),
        Catch::Matchers::ContainsSubstring("dense kernel path"));

    options = KernelOptions{};
    options.bins = 64u;
    options.binPath = BinPath::hash;
    options.scatter = ScatterChoice::direct;
    CHECK_THROWS_WITH(
        graphFlow::particle::runKernelAxes(device, parameters, options),
        Catch::Matchers::ContainsSubstring("dense-only"));

    options = KernelOptions{};
    options.bins = 4096u;
    options.binPath = BinPath::hash;
    parameters.tileSize = graphFlow::Config::hashBinTableTileSize + 1u;
    CHECK_THROWS_WITH(
        graphFlow::particle::runKernelAxes(device, parameters, options),
        Catch::Matchers::ContainsSubstring("tile-size"));

    options = KernelOptions{};
    options.bins = 64u;
    options.distribution = Distribution::sparse;
    parameters.tileSize = 128u;
    CHECK_THROWS_WITH(
        graphFlow::particle::runKernelAxes(device, parameters, options),
        Catch::Matchers::ContainsSubstring("sparse requires"));
}
