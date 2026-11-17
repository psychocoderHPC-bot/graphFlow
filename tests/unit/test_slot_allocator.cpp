/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/core/SlotAllocator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

TEST_CASE("slot arena hands out sequential ranges and tracks the free slots", "[slot][arena]")
{
    graphFlow::SlotArena arena{10u};
    CHECK(arena.numSlots() == 10u);
    CHECK(arena.freeSlots() == 10u);
    CHECK(arena.liveSlots() == 0u);
    CHECK(arena.liveRangeCount() == 0u);
    CHECK(arena.largestFreeRange() == 10u);
    CHECK(arena.freeRanges().size() == 1u);
    CHECK(arena.freeRanges().front().start == 0u);
    CHECK(arena.freeRanges().front().length == 10u);

    std::uint64_t const first = arena.allocate(3u);
    std::uint64_t const second = arena.allocate(4u);
    std::uint64_t const third = arena.allocate(3u);
    CHECK(first == 0u);
    CHECK(second == 3u);
    CHECK(third == 7u);
    CHECK(arena.freeSlots() == 0u);
    CHECK(arena.liveSlots() == 10u);
    CHECK(arena.liveRangeCount() == 3u);
    CHECK(arena.largestFreeRange() == 0u);
    CHECK_FALSE(arena.isFree(0u));
    CHECK_FALSE(arena.isFree(9u));
    CHECK_THROWS_AS(arena.allocate(1u), std::runtime_error);

    std::vector<graphFlow::SlotArena::Range> const live = arena.liveRanges();
    REQUIRE(live.size() == 3u);
    CHECK(live[0].start == 0u);
    CHECK(live[0].length == 3u);
    CHECK(live[1].start == 3u);
    CHECK(live[1].length == 4u);
    CHECK(live[2].start == 7u);
    CHECK(live[2].length == 3u);
    CHECK_NOTHROW(arena.verify());

    arena.release(first, 3u);
    arena.release(second, 4u);
    arena.release(third, 3u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("slot arena reuses a middle range and coalesces adjacent free ranges", "[slot][arena]")
{
    graphFlow::SlotArena arena{10u};
    std::uint64_t const first = arena.allocate(3u);
    std::uint64_t const middle = arena.allocate(4u);
    std::uint64_t const last = arena.allocate(3u);

    // Releasing a middle range makes exactly that range reusable.
    arena.release(middle, 4u);
    REQUIRE(arena.freeRanges().size() == 1u);
    CHECK(arena.freeRanges().front().start == 3u);
    CHECK(arena.freeRanges().front().length == 4u);
    CHECK(arena.isFree(3u));
    CHECK(arena.isFree(6u));
    CHECK_FALSE(arena.isFree(2u));
    CHECK_FALSE(arena.isFree(7u));

    std::uint64_t const reused = arena.allocate(4u);
    CHECK(reused == middle);
    CHECK_THROWS_AS(arena.allocate(4u), std::runtime_error);

    arena.release(reused, 4u);
    arena.release(first, 3u);
    // Adjacent free ranges merge into one, so a 7-slot request fits again.
    REQUIRE(arena.freeRanges().size() == 1u);
    CHECK(arena.freeRanges().front().start == 0u);
    CHECK(arena.freeRanges().front().length == 7u);
    CHECK(arena.largestFreeRange() == 7u);

    std::uint64_t const merged = arena.allocate(7u);
    CHECK(merged == 0u);
    arena.release(merged, 7u);
    arena.release(last, 3u);

    // A full release returns the arena to its initial single-range state.
    CHECK_NOTHROW(arena.verifyReleased());
    CHECK(arena.freeRanges().size() == 1u);
    CHECK(arena.freeRanges().front().start == 0u);
    CHECK(arena.freeRanges().front().length == 10u);
}

TEST_CASE("slot arena rejects a double free and an overlapping release", "[slot][arena]")
{
    graphFlow::SlotArena arena{8u};
    std::uint64_t const range = arena.allocate(4u);
    CHECK(range == 0u);

    // Freeing a live range twice is caught, and the arena state is untouched.
    arena.release(range, 4u);
    CHECK_THROWS_AS(arena.release(range, 4u), std::logic_error);
    CHECK(arena.freeSlots() == 8u);
    CHECK_NOTHROW(arena.verifyReleased());

    // A partial, shifted or oversized release of a live range is an overlap and is rejected.
    std::uint64_t const live = arena.allocate(4u);
    CHECK(live == 0u);
    CHECK_THROWS_AS(arena.release(1u, 2u), std::logic_error);
    CHECK_THROWS_AS(arena.release(0u, 3u), std::logic_error);
    CHECK_THROWS_AS(arena.release(0u, 5u), std::logic_error);
    CHECK_THROWS_AS(arena.release(2u, 4u), std::logic_error);
    CHECK(arena.liveSlots() == 4u);
    CHECK(arena.liveRangeCount() == 1u);
    CHECK_NOTHROW(arena.verify());

    // A shifted range that would extend past the live range is rejected as well.
    std::uint64_t const tail = arena.allocate(4u);
    CHECK(tail == 4u);
    arena.release(tail, 4u);
    arena.release(live, 4u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("slot arena detects a leak at teardown", "[slot][arena]")
{
    graphFlow::SlotArena arena{4u};
    std::uint64_t const first = arena.allocate(2u);
    std::uint64_t const second = arena.allocate(2u);

    // Still-live ranges are a leak even though the arena itself is consistent.
    CHECK_NOTHROW(arena.verify());
    CHECK_THROWS_AS(arena.verifyReleased(), std::logic_error);

    arena.release(first, 2u);
    CHECK_THROWS_AS(arena.verifyReleased(), std::logic_error);

    arena.release(second, 2u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("slot arena allocates single slots and keeps the arena fully tiled", "[slot][arena]")
{
    // One slot is the fragmentation worst case: every chunk owns exactly one slot.
    constexpr std::uint64_t slots = 5u;
    graphFlow::SlotArena arena{slots};
    std::vector<std::uint64_t> starts;
    for(std::uint64_t i = 0u; i < slots; ++i)
    {
        std::uint64_t const start = arena.allocate(1u);
        starts.push_back(start);
        CHECK_FALSE(arena.isFree(start));
        CHECK_NOTHROW(arena.verify());
    }
    CHECK(starts == std::vector<std::uint64_t>{0u, 1u, 2u, 3u, 4u});
    CHECK(arena.liveRangeCount() == slots);
    CHECK_THROWS_AS(arena.allocate(1u), std::runtime_error);

    // Free every second slot, then request a two-slot range: the hole is too small, so the
    // first-fit allocator must reject it instead of handing out an overlapping range.
    arena.release(starts[1], 1u);
    arena.release(starts[3], 1u);
    CHECK(arena.freeSlots() == 2u);
    CHECK(arena.largestFreeRange() == 1u);
    CHECK_THROWS_AS(arena.allocate(2u), std::runtime_error);
    CHECK_NOTHROW(arena.verify());

    // A single-slot request reuses the lowest free range.
    CHECK(arena.allocate(1u) == starts[1]);
    arena.release(starts[1], 1u);
    arena.release(starts[0], 1u);
    arena.release(starts[2], 1u);
    arena.release(starts[4], 1u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("slot arena rejects invalid requests and above-cap arenas", "[slot][arena]")
{
    graphFlow::SlotArena arena{5u};
    CHECK_THROWS_AS(arena.allocate(0u), std::invalid_argument);
    CHECK_THROWS_AS(arena.allocate(6u), std::invalid_argument);
    CHECK_THROWS_AS(arena.release(0u, 0u), std::invalid_argument);
    CHECK_THROWS_AS(arena.isFree(5u), std::invalid_argument);
    CHECK_THROWS_AS((graphFlow::SlotArena{0u}), std::invalid_argument);

    // The kernels index slots with 32 bits, so the arena cannot be wider than that.
    constexpr std::uint64_t tooWide = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1u;
    CHECK_THROWS_AS((graphFlow::SlotArena{tooWide}), std::overflow_error);

    std::uint64_t const whole = arena.allocate(5u);
    CHECK(whole == 0u);
    CHECK(arena.liveSlots() == 5u);
}
