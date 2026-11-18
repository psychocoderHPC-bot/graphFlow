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

TEST_CASE("a fragmented general free list cannot block a full-chunk allocation", "[slot][arena][reserve]")
{
    // The full-chunk class bumps down from the tail of a 128-slot arena; the general class
    // first-fits from the front.
    constexpr std::uint64_t block = 8u;
    graphFlow::SlotArena arena{128u, block};
    CHECK(arena.fullChunkSize() == block);
    CHECK(arena.fullRegionStart() == 128u);
    CHECK(arena.fullGap() == 128u);
    CHECK(arena.generalHighWater() == 0u);
    CHECK(arena.freeFullChunks() == 0u);

    // Three output ranges own the top 24 slots. They are invisible to the general free list, so
    // the 104 slots below them are what a stream of small input chunks can fragment.
    std::uint64_t const top = arena.allocateFull();
    std::uint64_t const middle = arena.allocateFull();
    std::uint64_t const bottom = arena.allocateFull();
    CHECK(top == 120u);
    CHECK(middle == 112u);
    CHECK(bottom == 104u);
    CHECK(arena.fullRegionStart() == 104u);

    std::vector<std::uint64_t> singles;
    for(std::uint64_t i = 0u; i < 104u; ++i)
        singles.push_back(arena.allocate(1u));
    for(std::uint64_t i = 0u; i < singles.size(); i += 2u)
        arena.release(singles[i], 1u);
    CHECK(arena.generalHighWater() == 104u);
    CHECK(arena.fullGap() == 0u);
    CHECK(arena.freeSlots() == 52u);
    CHECK(arena.largestFreeRange() == 1u); // every general hole is one slot wide
    // The general allocator cannot serve an 8-slot request from that free list.
    CHECK_THROWS_AS(arena.allocate(block), std::runtime_error);
    CHECK_NOTHROW(arena.verify());

    // A released output range is recycled by the class even though the general free list is
    // fragmented, so the C-sized allocation succeeds while the general one cannot.
    arena.release(middle, block);
    CHECK(arena.freeFullChunks() == 1u);
    std::uint64_t const reused = arena.allocateFull();
    CHECK(reused == middle);
    CHECK(arena.liveFullChunks() == 3u);
    CHECK_FALSE(arena.isFree(reused));
    CHECK_NOTHROW(arena.verify());

    // The mutation that removes the guarantee: the same fragmenting stream on an arena without
    // the class cannot serve the request and fails with the general first-fit error.
    graphFlow::SlotArena generalOnly{128u};
    for(std::uint64_t i = 0u; i < 128u; ++i)
        (void) generalOnly.allocate(1u);
    for(std::uint64_t i = 0u; i < 128u; i += 2u)
        generalOnly.release(i, 1u);
    std::string removed;
    try
    {
        [[maybe_unused]] std::uint64_t const blocked = generalOnly.allocate(block);
    }
    catch(std::runtime_error const& error)
    {
        removed = error.what();
    }
    CHECK(removed.find("no contiguous range of 8 slots is free") != std::string::npos);

    // With the class and no recycled range the request fails only when the arena's top is
    // genuinely consumed; the message names the request and the tested bound.
    std::string exhausted;
    try
    {
        [[maybe_unused]] std::uint64_t const blocked = arena.allocateFull();
    }
    catch(std::runtime_error const& error)
    {
        exhausted = error.what();
    }
    CHECK(
        exhausted.find("no full-chunk range of 8 slots is free above the general high-water mark")
        != std::string::npos);

    arena.release(reused, block);
    arena.release(bottom, block);
    arena.release(top, block);
    CHECK(arena.fullRegionStart() == 128u);
    for(std::uint64_t i = 1u; i < singles.size(); i += 2u)
        arena.release(singles[i], 1u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("the full-chunk class recycles a released middle range", "[slot][arena][reserve]")
{
    graphFlow::SlotArena arena{40u, 8u};
    std::uint64_t const top = arena.allocateFull();
    std::uint64_t const middle = arena.allocateFull();
    std::uint64_t const bottom = arena.allocateFull();
    CHECK(top == 32u);
    CHECK(middle == 24u);
    CHECK(bottom == 16u);

    // Releasing a range that is not at the bump cursor keeps it in the class free list, so the
    // next full-chunk request reuses exactly that start instead of bumping lower.
    arena.release(middle, 8u);
    CHECK(arena.freeFullChunks() == 1u);
    CHECK(arena.liveFullChunks() == 2u);
    std::uint64_t const reused = arena.allocateFull();
    CHECK(reused == middle);
    CHECK_NOTHROW(arena.verify());

    arena.release(reused, 8u);
    // A range at the cursor merges back into the general free list, so releasing everything
    // restores the single [0, 40) range.
    arena.release(bottom, 8u);
    arena.release(top, 8u);
    CHECK(arena.fullRegionStart() == 40u);
    CHECK_NOTHROW(arena.verifyReleased());

    // A general 8-slot request is served from the general free list at the front.
    std::uint64_t const general = arena.allocate(8u);
    CHECK(general == 0u);
    CHECK(arena.freeFullChunks() == 0u);
    arena.release(general, 8u);
    CHECK_NOTHROW(arena.verifyReleased());
}

TEST_CASE("slot arena rejects an invalid full-chunk class", "[slot][arena][reserve]")
{
    // A full-chunk range must fit the arena.
    CHECK_THROWS_AS((graphFlow::SlotArena{4u, 8u}), std::invalid_argument);
    CHECK_THROWS_AS((graphFlow::SlotArena{0u, 8u}), std::invalid_argument);

    // Without the class a full-chunk request is a loud misconfiguration, not a silent fallback.
    graphFlow::SlotArena plain{16u};
    CHECK_THROWS_AS(plain.allocateFull(), std::logic_error);
    CHECK(plain.freeFullChunks() == 0u);
}
