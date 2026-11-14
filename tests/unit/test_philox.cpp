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
    std::vector<std::uint32_t> sequence(graphFlow::rng::Engine engine, std::size_t count)
    {
        std::vector<std::uint32_t> values;
        values.reserve(count);
        for(std::size_t i = 0u; i < count; ++i)
            values.push_back(engine());
        return values;
    }
} // namespace

TEST_CASE("makeEngine is reproducible for one stream position", "[rng]")
{
    for(std::uint64_t const seed : {0ull, 1ull, 7ull, 42ull, 0x1234'5678'9ABC'DEF0ull})
    {
        for(std::uint64_t const id : {0ull, 1ull, 2ull, 5ull, 1000ull, 0xFFFF'FFFFull})
        {
            for(std::uint32_t const event : {0u, 1u, 7u, 0xFFFFu})
            {
                CAPTURE(seed, id, event);
                auto const first = graphFlow::rng::makeEngine(seed, id, event, 6u);
                auto const second = graphFlow::rng::makeEngine(seed, id, event, 6u);
                CHECK(sequence(first, 8u) == sequence(second, 8u));
            }
        }
    }
}

TEST_CASE("stream keys are unique within the packed limits", "[rng]")
{
    // The packed position and the splitmix64 finalizer are bijections, so every distinct
    // (id, event, purpose) position within the documented bit budget owns a distinct key.
    std::vector<std::uint64_t> keys;
    for(std::uint64_t id = 0u; id < 64u; ++id)
        for(std::uint64_t event = 0u; event < 64u; ++event)
            for(std::uint32_t purpose = 0u; purpose < 32u; ++purpose)
                keys.push_back(graphFlow::rng::checkedStreamKey(42u, id, event, purpose));

    std::sort(keys.begin(), keys.end());
    CHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
    CHECK(keys.size() == 64u * 64u * 32u);
}

TEST_CASE("engines own disjoint streams", "[rng]")
{
    auto const base = sequence(graphFlow::rng::makeEngine(42u, 7u, 3u, 6u), 16u);
    auto const otherId = sequence(graphFlow::rng::makeEngine(42u, 8u, 3u, 6u), 16u);
    auto const otherEvent = sequence(graphFlow::rng::makeEngine(42u, 7u, 4u, 6u), 16u);
    auto const otherPurpose = sequence(graphFlow::rng::makeEngine(42u, 7u, 3u, 7u), 16u);
    auto const same = sequence(graphFlow::rng::makeEngine(42u, 7u, 3u, 6u), 16u);

    CHECK(base == same);
    CHECK(base != otherId);
    CHECK(base != otherEvent);
    CHECK(base != otherPurpose);
    // No element of two different positions may coincide for this fixed sample.
    for(std::size_t i = 0u; i < base.size(); ++i)
    {
        CHECK(base[i] != otherId[i]);
        CHECK(base[i] != otherEvent[i]);
        CHECK(base[i] != otherPurpose[i]);
    }
}

TEST_CASE("makeEngine never uses the pinned constructor offset path", "[rng]")
{
    // The key is set at construction with subsequence and offset zero. Re-constructing the same
    // key directly must therefore match, and changing the event must change the first draw.
    for(std::uint64_t const seed : {0ull, 42ull, 0xDEAD'BEEF'CAFE'F00Dull})
    {
        for(std::uint64_t const id : {0ull, 1ull, 7ull, 0xABCD'EF01ull})
        {
            for(std::uint32_t const purpose : {0u, 1u, 6u, 0xFFFFu})
            {
                CAPTURE(seed, id, purpose);
                std::uint64_t const key = graphFlow::rng::checkedStreamKey(seed, id, 0u, purpose);
                graphFlow::rng::Engine direct{key, 0u, 0u};
                graphFlow::rng::Engine made = graphFlow::rng::makeEngine(seed, id, 0u, purpose);
                CHECK(direct() == made());
            }
        }
    }
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 0u, 6u)() != graphFlow::rng::makeEngine(42u, 7u, 1u, 6u)());
}

TEST_CASE("checked stream keys reject positions outside the bit budget", "[rng]")
{
    using graphFlow::rng::checkedStreamKey;
    using graphFlow::rng::maxStreamEvent;
    using graphFlow::rng::maxStreamId;
    using graphFlow::rng::maxStreamPurpose;

    CHECK_NOTHROW(checkedStreamKey(42u, maxStreamId, maxStreamEvent, maxStreamPurpose));
    CHECK_THROWS_AS(checkedStreamKey(42u, maxStreamId + 1u, 0u, 0u), std::overflow_error);
    CHECK_THROWS_AS(checkedStreamKey(42u, 0u, maxStreamEvent + 1u, 0u), std::overflow_error);
    CHECK_THROWS_AS(checkedStreamKey(42u, 0u, 0u, maxStreamPurpose + 1u), std::overflow_error);
}

TEST_CASE("scalar oracle uses an independent stream mapping", "[rng]")
{
    // ScalarModel includes no production RNG header; compare its locally implemented packing and
    // mixer with the production wrapper over a key matrix. This catches a mapping change on either
    // side.
    for(std::uint64_t const seed : {0ull, 7ull, 42ull})
    {
        for(std::uint64_t const id : {0ull, 1ull, 2ull, 7ull, 1000ull})
        {
            for(std::uint32_t visit = 0u; visit < 4u; ++visit)
            {
                for(std::uint32_t purpose = 0u; purpose < 8u; ++purpose)
                {
                    CAPTURE(seed, id, visit, purpose);
                    auto const expected = graphFlow::test::scalarRng::engineFor(seed, id, visit, purpose);
                    auto const actual = graphFlow::rng::makeEngine(seed, id, visit, purpose);
                    CHECK(sequence(expected, 4u) == sequence(actual, 4u));
                }
            }
        }
    }
}

TEST_CASE("fixed-seed engine outputs are pinned", "[rng]")
{
    // Produced with a fresh keyed engine at offset zero after the RNG API change. The values pin
    // the packing/mixer and the zero-offset construction, not an alpaka constructor offset.
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 0u, 0u)() == 0xd0fd'0cc6u);
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 0u, 2u)() == 0xfda9'155eu);
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 0u, 4u)() == 0x6f01'57dfu);
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 1u, 4u)() == 0xab9f'ef20u);
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 0u, 6u)() == 0xe0cb'a29eu);
    CHECK(graphFlow::rng::makeEngine(42u, 7u, 2u, 6u)() == 0x752f'fe28u);
}

TEST_CASE("UniformReal converts the keyed engine deterministically", "[rng]")
{
    using interval = alpaka::rand::interval::CO;
    graphFlow::rng::Engine engine = graphFlow::rng::makeEngine(42u, 7u, 2u, 6u);
    auto const first = alpaka::rand::distribution::UniformReal<double, interval>{}(engine);
    auto const second = alpaka::rand::distribution::UniformReal<double, interval>{}(engine);

    graphFlow::rng::Engine repeat = graphFlow::rng::makeEngine(42u, 7u, 2u, 6u);
    CHECK(alpaka::rand::distribution::UniformReal<double, interval>{}(repeat) == first);
    CHECK(alpaka::rand::distribution::UniformReal<double, interval>{}(repeat) == second);
    CHECK(first >= 0.0);
    CHECK(first < 1.0);
    CHECK(second >= 0.0);
    CHECK(second < 1.0);
}
