/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ScalarModel.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
    std::uint32_t advance(std::uint64_t seed, std::uint64_t subsequence, std::uint64_t offset)
    {
        alpaka::rand::engine::Philox4x32x10 engine{seed, subsequence, 0u};
        for(std::uint64_t i = 0u; i < offset; ++i)
            engine();
        return engine();
    }

    // Compile-time purpose checks. An invalid purpose must not compile; these instantiate both the
    // three-purpose particle and the four-purpose photon mapping.
    static_assert(graphFlow::rng::checkedDecisionOffset<graphFlow::rng::purposeFilter>(0u) == 4u);
    static_assert(graphFlow::rng::checkedDecisionOffset<graphFlow::rng::purposeProcessB>(3u) == 48u);
    static_assert(
        graphFlow::rng::checkedDecisionOffset<
            graphFlow::rng::purposeAzimuth,
            graphFlow::rng::photonPurposesPerVisit>(1u)
        == 32u);
} // namespace

TEST_CASE("drawAt equals advancing a fresh engine", "[rng]")
{
    // The example derives every decision from an absolute draw offset, so this equivalence is the
    // reproducibility contract of the whole RNG mapping.
    for(std::uint64_t const seed : {0ull, 1ull, 7ull, 42ull, 0x123456789ABCDEF0ull})
    {
        for(std::uint64_t const id : {0ull, 1ull, 2ull, 5ull, 1000ull, 0xFFFFFFFFull})
        {
            for(std::uint64_t offset = 0ull; offset <= 130ull; ++offset)
            {
                CAPTURE(seed, id, offset);
                CHECK(graphFlow::rng::drawAt(seed, id, offset) == advance(seed, id, offset));
            }
        }
    }
}

TEST_CASE("pinned Philox constructor offset is not equivalent to advancing", "[rng]")
{
    // Pinned behaviour of the single-value engine, recorded so that an upstream fix is noticed:
    //  - skip() sets the within-block position and advanceState() resets it, so offsets 1..3 alias
    //    offset 0,
    //  - skip4() derives its carry words from counter[0] < temp[1] / counter[0] < temp[2], which
    //    perturbs counter word 3 for a nonzero subsequence even at offset 0.
    // Consequence: the raw constructor offset must not be used for a key-based mapping.
    // graphFlow::rng::drawAt avoids it by advancing a fresh engine.
    // If this test starts failing, the pinned alpaka changed; revisit the wrapper and this note.
    for(std::uint64_t const offset : {1ull, 2ull, 3ull})
    {
        CAPTURE(offset);
        alpaka::rand::engine::Philox4x32x10 zero{42u, 7u, 0u};
        alpaka::rand::engine::Philox4x32x10 direct{42u, 7u, offset};
        CHECK(direct() == zero());
    }
    // A whole block jump with a nonzero subsequence diverges from advancing a fresh engine.
    alpaka::rand::engine::Philox4x32x10 block{42u, 1u, 4u};
    CHECK(block() != advance(42u, 1u, 4u));
    // With subsequence 0 the pinned arithmetic happens to agree for whole blocks.
    alpaka::rand::engine::Philox4x32x10 block0{42u, 0u, 4u};
    CHECK(block0() == advance(42u, 0u, 4u));
}

TEST_CASE("decision offsets are the documented 4*(1+3*visit+purpose) map", "[rng]")
{
    using graphFlow::rng::decisionOffset;

    CHECK(decisionOffset(0u, graphFlow::rng::purposeFilter) == 4u);
    CHECK(decisionOffset(0u, graphFlow::rng::purposeProcessA) == 8u);
    CHECK(decisionOffset(0u, graphFlow::rng::purposeProcessB) == 12u);
    CHECK(decisionOffset(1u, graphFlow::rng::purposeFilter) == 16u);
    CHECK(decisionOffset(1u, graphFlow::rng::purposeProcessA) == 20u);
    CHECK(decisionOffset(1u, graphFlow::rng::purposeProcessB) == 24u);
    CHECK(decisionOffset(3u, graphFlow::rng::purposeProcessB) == 4u * (1u + 9u + 2u));

    // The property draw owns offset 0 and must differ from the first filter draw.
    CHECK(graphFlow::rng::propertyDraw(42u, 7u) != graphFlow::rng::draw(42u, 7u, 0u, graphFlow::rng::purposeFilter));
}

TEST_CASE("draw positions are independent of the caller", "[rng]")
{
    // Same key -> same value, no matter in which order other particles are evaluated.
    std::vector<std::uint32_t> forward;
    std::vector<std::uint32_t> backward;
    for(std::uint64_t id = 0u; id < 32u; ++id)
        forward.push_back(graphFlow::rng::draw(7u, id, 2u, graphFlow::rng::purposeProcessA));
    for(std::uint64_t id = 32u; id-- > 0u;)
        backward.push_back(graphFlow::rng::draw(7u, id, 2u, graphFlow::rng::purposeProcessA));

    std::reverse(backward.begin(), backward.end());
    CHECK(forward == backward);
}

TEST_CASE("production draw mapping matches pinned literal outputs", "[rng]")
{
    // Independent pin of the (seed, id, visit, purpose) -> 32 bit draw mapping. The values were
    // produced by constructing a fresh Philox4x32x10 at offset zero and advancing locally; they do
    // not depend on graphFlow's wrapper. A shared offset-mapping defect cannot hide here.
    CHECK(graphFlow::rng::propertyDraw(42u, 7u) == 0xe9f1f921u);
    CHECK(graphFlow::rng::draw(42u, 7u, 0u, graphFlow::rng::purposeFilter) == 0x2f001ca4u);
    CHECK(graphFlow::rng::draw(42u, 7u, 0u, graphFlow::rng::purposeProcessA) == 0x8153eeefu);
    CHECK(graphFlow::rng::draw(42u, 7u, 1u, graphFlow::rng::purposeProcessA) == 0xb7bfefb0u);
    CHECK(graphFlow::rng::draw(42u, 7u, 0u, graphFlow::rng::purposeProcessB) == 0x105f9ddau);
    CHECK(graphFlow::rng::draw(42u, 7u, 2u, graphFlow::rng::purposeProcessB) == 0x1e7bf881u);
}

TEST_CASE("scalar oracle uses an independent Philox mapping", "[rng]")
{
    // ScalarModel includes no production RNG header; compare its locally implemented mapping with
    // the production wrapper over a key matrix. This catches a mapping change on either side.
    for(std::uint64_t const seed : {0ull, 7ull, 42ull})
    {
        for(std::uint64_t const id : {0ull, 1ull, 2ull, 7ull, 1000ull})
        {
            CHECK(graphFlow::test::scalarRng::propertyDraw(seed, id) == graphFlow::rng::propertyDraw(seed, id));
            for(std::uint32_t visit = 0u; visit < 4u; ++visit)
            {
                for(std::uint32_t purpose = 0u; purpose < 3u; ++purpose)
                {
                    CAPTURE(seed, id, visit, purpose);
                    CHECK(
                        graphFlow::test::scalarRng::draw(seed, id, visit, purpose)
                        == graphFlow::rng::draw(seed, id, visit, purpose));
                }
            }
        }
    }
}

TEST_CASE("three and four purpose visits own disjoint draw blocks", "[rng]")
{
    auto checkMapping = [](std::uint32_t purposes)
    {
        std::vector<std::uint64_t> offsets;
        offsets.reserve(8u * purposes);
        for(std::uint32_t visit = 0u; visit < 8u; ++visit)
        {
            for(std::uint32_t purpose = 0u; purpose < purposes; ++purpose)
            {
                std::uint64_t const offset = graphFlow::rng::decisionOffset(visit, purpose, purposes);
                CAPTURE(purposes, visit, purpose);
                CHECK(offset == 4ull * (1ull + static_cast<std::uint64_t>(purposes) * visit + purpose));
                offsets.push_back(offset);
            }
        }

        CHECK(offsets.front() == 4u);
        std::sort(offsets.begin(), offsets.end());
        CHECK(std::adjacent_find(offsets.begin(), offsets.end()) == offsets.end());
        // Every purpose owns one four-word Philox block, so neighbouring positions cannot overlap.
        for(std::size_t i = 1u; i < offsets.size(); ++i)
            CHECK(offsets[i] - offsets[i - 1u] >= 4u);
    };

    checkMapping(graphFlow::rng::purposesPerVisit);
    checkMapping(graphFlow::rng::photonPurposesPerVisit);

    using graphFlow::rng::decisionOffset;
    using graphFlow::rng::photonPurposesPerVisit;
    CHECK(decisionOffset(0u, graphFlow::rng::purposeFlight, photonPurposesPerVisit) == 4u);
    CHECK(decisionOffset(0u, graphFlow::rng::purposeCollisionKind, photonPurposesPerVisit) == 8u);
    CHECK(decisionOffset(0u, graphFlow::rng::purposeCosine, photonPurposesPerVisit) == 12u);
    CHECK(decisionOffset(0u, graphFlow::rng::purposeAzimuth, photonPurposesPerVisit) == 16u);
    CHECK(decisionOffset(1u, graphFlow::rng::purposeAzimuth, photonPurposesPerVisit) == 32u);
}

TEST_CASE("checked draw offsets reject overflow and out-of-range purposes", "[rng]")
{
    using graphFlow::rng::checkedDecisionOffset;

    CHECK(checkedDecisionOffset(0u, graphFlow::rng::purposeProperty) == 4u);
    CHECK(checkedDecisionOffset(3u, graphFlow::rng::purposeProcessB) == 48u);
    CHECK_THROWS_AS(checkedDecisionOffset(0u, graphFlow::rng::photonPurposesPerVisit), std::invalid_argument);
    CHECK_THROWS_AS(checkedDecisionOffset(0u, 0u, 0u), std::invalid_argument);

    constexpr std::uint64_t maxOffset = std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t largestBlock = maxOffset / 4u;
    std::uint64_t const visit = (largestBlock - 1u) / graphFlow::rng::purposesPerVisit;
    CHECK(checkedDecisionOffset(visit, 0u) <= maxOffset);
    CHECK_THROWS_AS(checkedDecisionOffset(largestBlock, 0u), std::overflow_error);
    CHECK_THROWS_AS(checkedDecisionOffset(maxOffset, 0u), std::overflow_error);
}

TEST_CASE("fixed-seed photon draws reproduce the independent mapping", "[rng]")
{
    constexpr std::uint64_t seed = 42u;
    constexpr std::uint64_t id = 7u;
    constexpr std::uint32_t visit = 2u;
    using graphFlow::rng::photonPurposesPerVisit;

    std::array<std::uint32_t, photonPurposesPerVisit> first{};
    std::array<std::uint32_t, photonPurposesPerVisit> second{};
    for(std::uint32_t purpose = 0u; purpose < photonPurposesPerVisit; ++purpose)
    {
        first[purpose] = graphFlow::rng::draw(seed, id, visit, purpose, photonPurposesPerVisit);
        second[purpose] = graphFlow::rng::draw(seed, id, visit, purpose, photonPurposesPerVisit);
        CAPTURE(purpose);
        CHECK(
            first[purpose]
            == advance(seed, id, graphFlow::rng::decisionOffset(visit, purpose, photonPurposesPerVisit)));
    }
    CHECK(first == second);
    CHECK(
        graphFlow::rng::draw<graphFlow::rng::purposeFlight, photonPurposesPerVisit>(seed, id, visit)
        == first[0]);
    CHECK(
        graphFlow::rng::draw<graphFlow::rng::purposeCollisionKind, photonPurposesPerVisit>(seed, id, visit)
        == first[1]);
    CHECK(
        graphFlow::rng::draw<graphFlow::rng::purposeCosine, photonPurposesPerVisit>(seed, id, visit) == first[2]);
    CHECK(
        graphFlow::rng::draw<graphFlow::rng::purposeAzimuth, photonPurposesPerVisit>(seed, id, visit)
        == first[3]);
}
