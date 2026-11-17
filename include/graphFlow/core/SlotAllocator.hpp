/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphFlow
{
    /** Host-side allocator of contiguous slot ranges inside one flat arena.
     *
     * The slot-arena storage gives every chunk a contiguous range of exactly its committed fill
     * instead of the fixed row `index * chunkCapacity`, so a one-particle fragment owns one slot
     * rather than a full chunk. This type owns the matching free list: disjoint free ranges,
     * initially `[0, numSlots)`, from which `allocate()` carves the first fitting range and into
     * which `release()` returns it while coalescing adjacent neighbours.
     *
     * The allocator is host-only and never touches device memory, so it is unit-testable without a
     * backend. The free list is the authoritative ownership record: a double free, an overlapping
     * or partial release, a zero-length request and a lost range throw instead of silently handing
     * two live chunks the same slots. `verify()` re-derives ownership from both sides and proves
     * that the free and live ranges tile the whole arena exactly once.
     *
     * Kernels index slots with 32-bit arithmetic, so an arena larger than `UINT32_MAX` slots is
     * rejected at construction. The fixed layout gets the same cap from `checkedPoolSlotCount`.
     *
     * The allocator is a first-fit range allocator, not a block allocator: a request of any length
     * up to `numSlots` is served when a free range that long exists, which is what lets output
     * chunks ask for a full `chunkCapacity` and source chunks ask for their exact fill.
     */
    class SlotArena
    {
    public:
        /** Sentinel meaning "no slot range"; a released chunk index stores it. */
        static constexpr std::uint64_t noSlot = std::numeric_limits<std::uint64_t>::max();

        /** One contiguous slot range `[start, start + length)`. */
        struct Range
        {
            std::uint64_t start = 0u;
            std::uint64_t length = 0u;
        };

        explicit SlotArena(std::uint64_t numSlots) : m_numSlots{numSlots}
        {
            if(numSlots == 0u)
                throw std::invalid_argument{"graphFlow::SlotArena: numSlots must be non-zero"};
            if(numSlots > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow::SlotArena: the arena exceeds the 32-bit slot index"};
            m_free.push_back(Range{0u, numSlots});
        }

        /** Carve the first free range of exactly `count` slots and return its start.
         *
         * Throws `std::invalid_argument` for a zero-length or oversized request and
         * `std::runtime_error` when no free range is long enough, which mirrors the fixed pool's
         * exhaustion behavior.
         */
        std::uint64_t allocate(std::uint64_t count)
        {
            if(count == 0u)
                throw std::invalid_argument{"graphFlow::SlotArena: a slot range must be non-zero"};
            if(count > m_numSlots)
                throw std::invalid_argument{
                    "graphFlow::SlotArena: a range of " + std::to_string(count) + " slots exceeds the arena of "
                    + std::to_string(m_numSlots) + " slots"};
            for(std::size_t i = 0u; i < m_free.size(); ++i)
            {
                Range const free = m_free[i];
                if(free.length < count)
                    continue;
                if(free.length == count)
                    m_free.erase(m_free.begin() + static_cast<std::ptrdiff_t>(i));
                else
                    m_free[i] = Range{free.start + count, free.length - count};
                auto const inserted = m_live.emplace(free.start, count);
                if(!inserted.second)
                    throw std::logic_error{"graphFlow::SlotArena: the free list handed out a live range"};
                return free.start;
            }
            throw std::runtime_error{
                "graphFlow::SlotArena: no contiguous range of " + std::to_string(count) + " slots is free"};
        }

        /** Return a range that `allocate()` handed out.
         *
         * A range that is not live at `start`, or whose length differs from the allocation, raises
         * `std::logic_error`: that is the double-free, partial-release and overlap case.
         */
        void release(std::uint64_t start, std::uint64_t count)
        {
            if(count == 0u)
                throw std::invalid_argument{"graphFlow::SlotArena: a slot range must be non-zero"};
            auto const live = m_live.find(start);
            if(live == m_live.end())
                throw std::logic_error{
                    "graphFlow::SlotArena: cannot release " + std::to_string(count) + " slots at "
                    + std::to_string(start) + "; the range is not live (double free, partial release or overlap)"};
            if(live->second != count)
                throw std::logic_error{
                    "graphFlow::SlotArena: the range at " + std::to_string(start) + " owns "
                    + std::to_string(live->second) + " slots, not " + std::to_string(count)};
            m_live.erase(live);
            insertFree(Range{start, count});
        }

        [[nodiscard]] std::uint64_t numSlots() const
        {
            return m_numSlots;
        }

        [[nodiscard]] std::uint64_t freeSlots() const
        {
            std::uint64_t slots = 0u;
            for(auto const& range : m_free)
                slots += range.length;
            return slots;
        }

        [[nodiscard]] std::uint64_t liveSlots() const
        {
            std::uint64_t slots = 0u;
            for(auto const& entry : m_live)
                slots += entry.second;
            return slots;
        }

        [[nodiscard]] std::uint32_t liveRangeCount() const
        {
            return static_cast<std::uint32_t>(m_live.size());
        }

        /** Longest free range; zero when the arena is fully allocated. */
        [[nodiscard]] std::uint64_t largestFreeRange() const
        {
            std::uint64_t largest = 0u;
            for(auto const& range : m_free)
                largest = range.length > largest ? range.length : largest;
            return largest;
        }

        /** True when `slot` is not owned by any live range. */
        [[nodiscard]] bool isFree(std::uint64_t slot) const
        {
            if(slot >= m_numSlots)
                throw std::invalid_argument{"graphFlow::SlotArena: slot index out of range"};
            for(auto const& range : m_free)
            {
                if(slot < range.start)
                    return false;
                if(slot < range.start + range.length)
                    return true;
            }
            return false;
        }

        /** Free ranges, ascending by start and coalesced. */
        [[nodiscard]] std::vector<Range> const& freeRanges() const
        {
            return m_free;
        }

        /** Live ranges, ascending by start. */
        [[nodiscard]] std::vector<Range> liveRanges() const
        {
            std::vector<Range> live;
            live.reserve(m_live.size());
            for(auto const& entry : m_live)
                live.push_back(Range{entry.first, entry.second});
            return live;
        }

        /** Assert the arena invariants; throws `std::logic_error` on the first violation.
         *
         * The check walks the free and live ranges together and proves three things: every range
         * is non-empty and inside the arena, every slot is covered exactly once (so no two live
         * ranges overlap and no slot is owned by two ranges), and adjacent free ranges are
         * coalesced. Cost is `O(n log n)` in the number of ranges, so it is a debug and test check,
         * not part of the allocation path.
         */
        void verify() const
        {
            struct Owned
            {
                std::uint64_t start;
                std::uint64_t length;
                bool free;
            };

            std::vector<Range> const live = liveRanges();
            std::vector<Owned> ranges;
            ranges.reserve(m_free.size() + live.size());
            for(auto const& range : m_free)
                ranges.push_back(Owned{range.start, range.length, true});
            for(auto const& range : live)
                ranges.push_back(Owned{range.start, range.length, false});
            std::sort(
                ranges.begin(),
                ranges.end(),
                [](Owned const& lhs, Owned const& rhs) { return lhs.start < rhs.start; });

            std::uint64_t expected = 0u;
            for(std::size_t i = 0u; i < ranges.size(); ++i)
            {
                Owned const& range = ranges[i];
                if(range.length == 0u)
                    throw std::logic_error{"graphFlow::SlotArena: a zero-length range is recorded"};
                if(range.start < expected)
                    throw std::logic_error{
                        "graphFlow::SlotArena: the range at " + std::to_string(range.start)
                        + " overlaps the preceding range"};
                if(range.start > expected)
                    throw std::logic_error{
                        "graphFlow::SlotArena: no range owns slots [" + std::to_string(expected) + ", "
                        + std::to_string(range.start) + ")"};
                if(range.start + range.length > m_numSlots)
                    throw std::logic_error{"graphFlow::SlotArena: a range extends past the arena"};
                if(i > 0u && range.free && ranges[i - 1u].free
                   && ranges[i - 1u].start + ranges[i - 1u].length == range.start)
                    throw std::logic_error{"graphFlow::SlotArena: adjacent free ranges are not coalesced"};
                expected = range.start + range.length;
            }
            if(expected != m_numSlots)
                throw std::logic_error{
                    "graphFlow::SlotArena: the arena is not fully tiled; " + std::to_string(expected) + " of "
                    + std::to_string(m_numSlots) + " slots are accounted for"};
        }

        /** Leak check: no live range may remain and the free list must be `[0, numSlots)` again. */
        void verifyReleased() const
        {
            verify();
            if(!m_live.empty())
                throw std::logic_error{
                    "graphFlow::SlotArena: " + std::to_string(m_live.size())
                    + " slot range(s) are still live at teardown"};
            if(m_free.size() != 1u || m_free.front().start != 0u || m_free.front().length != m_numSlots)
                throw std::logic_error{
                    "graphFlow::SlotArena: a full release must return the arena to its initial state"};
        }

    private:
        /** Insert a returned range and merge it with adjacent free neighbours. */
        void insertFree(Range range)
        {
            auto const at = std::lower_bound(
                m_free.begin(),
                m_free.end(),
                range.start,
                [](Range const& lhs, std::uint64_t start) { return lhs.start < start; });
            if(at != m_free.end() && at->start < range.start + range.length)
                throw std::logic_error{"graphFlow::SlotArena: a released range overlaps a free range"};
            if(at != m_free.begin() && std::prev(at)->start + std::prev(at)->length > range.start)
                throw std::logic_error{"graphFlow::SlotArena: a released range overlaps a free range"};
            auto inserted = m_free.insert(at, range);
            // Merge with the successor first: merging with the predecessor may erase it, which
            // would invalidate the successor iterator.
            auto next = std::next(inserted);
            if(next != m_free.end() && inserted->start + inserted->length == next->start)
            {
                inserted->length += next->length;
                m_free.erase(next);
            }
            if(inserted != m_free.begin())
            {
                auto previous = std::prev(inserted);
                if(previous->start + previous->length == inserted->start)
                {
                    previous->length += inserted->length;
                    m_free.erase(inserted);
                }
            }
        }

        std::uint64_t m_numSlots;
        std::vector<Range> m_free;
        std::map<std::uint64_t, std::uint64_t> m_live;
    };
} // namespace graphFlow
