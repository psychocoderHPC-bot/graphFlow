/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "ScalarModel.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
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
