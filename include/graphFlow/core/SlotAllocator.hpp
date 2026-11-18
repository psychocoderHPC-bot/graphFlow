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
     *
     * First-fit alone cannot serve a large aligned request once many small releases have chopped
     * the free list: total free slots may exceed the request while no single contiguous range is
     * long enough. Output chunks are the case that must not fail, because the scatter kernel
     * addresses every chunk of a bin's dense sequence at `j % chunkCapacity`, so its range must be
     * a full `chunkCapacity` wide. The arena therefore keeps a **full-chunk size class** that
     * bumps down from the tail while the general first-fit list grows from the front; the two
     * share one budget and meet under it. A full-chunk request never inspects the general free
     * list, so small-range fragmentation cannot block it: it needs `chunkCapacity` free slots
     * above the general high-water mark, and a released full-chunk range is recycled directly.
     *
     * Failure mode: a full-chunk request fails only when the general high-water mark has grown to
     * within one `chunkCapacity` of the full-chunk cursor, i.e. general allocations have consumed
     * the arena up to the top. That is a genuinely full arena, not fragmentation, and it raises a
     * loud `std::runtime_error` naming the request and the tested bound.
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

        /** Arena served by one first-fit free list; no full-chunk size class. */
        explicit SlotArena(std::uint64_t numSlots) : SlotArena{numSlots, 0u}
        {
        }

        /** Arena with a full-chunk size class taken from the tail.
         *
         * `fullChunkSize` is `chunkCapacity` (C). General requests (`allocate()`) are first-fit
         * from the front, full-chunk requests (`allocateFull()`) bump down from the tail, and the
         * two grow towards each other under one budget. A full-chunk request therefore never
         * inspects the general free list, so small-range fragmentation cannot block it: it only
         * needs `fullChunkSize` free slots above the general high-water mark. A released
         * full-chunk range is recycled by the next full-chunk request. A zero `fullChunkSize`
         * disables the class and `allocateFull()` reports that instead of silently degrading.
         */
        SlotArena(std::uint64_t numSlots, std::uint64_t fullChunkSize)
            : m_numSlots{numSlots}
            , m_fullChunkSize{fullChunkSize}
        {
            if(numSlots == 0u)
                throw std::invalid_argument{"graphFlow::SlotArena: numSlots must be non-zero"};
            if(numSlots > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow::SlotArena: the arena exceeds the 32-bit slot index"};
            if(fullChunkSize > numSlots)
                throw std::invalid_argument{
                    "graphFlow::SlotArena: a full-chunk range of " + std::to_string(fullChunkSize)
                    + " slots does not fit an arena of " + std::to_string(numSlots) + " slots"};
            m_fullCursor = numSlots;
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
                auto const inserted = m_live.emplace(free.start, Live{count, false});
                if(!inserted.second)
                    throw std::logic_error{"graphFlow::SlotArena: the free list handed out a live range"};
                if(free.start + count > m_generalHigh)
                    m_generalHigh = free.start + count;
                return free.start;
            }
            throw std::runtime_error{
                "graphFlow::SlotArena: no contiguous range of " + std::to_string(count) + " slots is free"};
        }

        /** Carve one full-chunk range from the tail and return its start.
         *
         * The general free list is not consulted, so fragmentation by small ranges cannot make
         * this fail. A released full-chunk range is reused first; otherwise the range is taken
         * from the top of the free arena above the general high-water mark. The only failure is a
         * genuinely full arena, which raises a `std::runtime_error` naming the request and the
         * available gap.
         */
        std::uint64_t allocateFull()
        {
            if(m_fullChunkSize == 0u)
                throw std::logic_error{
                    "graphFlow::SlotArena: the arena has no full-chunk size class, so a full-chunk request "
                    "cannot be served"};
            std::uint64_t start = 0u;
            if(!m_fullFree.empty())
            {
                start = m_fullFree.back();
                m_fullFree.pop_back();
            }
            else
            {
                if(m_fullCursor < m_fullChunkSize || m_fullCursor - m_fullChunkSize < m_generalHigh)
                    throw std::runtime_error{
                        "graphFlow::SlotArena: no full-chunk range of " + std::to_string(m_fullChunkSize)
                        + " slots is free above the general high-water mark of " + std::to_string(m_generalHigh)
                        + " in an arena of " + std::to_string(m_numSlots) + " slots"};
                start = takeTopOfGeneral(m_fullChunkSize);
            }
            auto const inserted = m_live.emplace(start, Live{m_fullChunkSize, true});
            if(!inserted.second)
                throw std::logic_error{"graphFlow::SlotArena: a released full-chunk range was handed out twice"};
            std::uint64_t const live = liveFullChunks();
            if(live > m_peakLiveFull)
                m_peakLiveFull = live;
            return start;
        }

        /** Return a range that `allocate()` or `allocateFull()` handed out.
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
            if(live->second.length != count)
                throw std::logic_error{
                    "graphFlow::SlotArena: the range at " + std::to_string(start) + " owns "
                    + std::to_string(live->second.length) + " slots, not " + std::to_string(count)};
            bool const full = live->second.full;
            m_live.erase(live);
            if(full)
                insertFullFree(start);
            else
                insertFree(Range{start, count});
        }

        [[nodiscard]] std::uint64_t numSlots() const
        {
            return m_numSlots;
        }

        /** Size of one full-chunk range; zero when the class is disabled. */
        [[nodiscard]] std::uint64_t fullChunkSize() const
        {
            return m_fullChunkSize;
        }

        /** First slot that belongs to the full-chunk class; equals `numSlots` when unused. */
        [[nodiscard]] std::uint64_t fullRegionStart() const
        {
            return m_fullCursor;
        }

        /** Highest slot ever handed out to a general request; the full class must stay above it. */
        [[nodiscard]] std::uint64_t generalHighWater() const
        {
            return m_generalHigh;
        }

        /** Released full-chunk ranges waiting for reuse. */
        [[nodiscard]] std::uint64_t freeFullChunks() const
        {
            return static_cast<std::uint64_t>(m_fullFree.size());
        }

        /** Live full-chunk ranges. */
        [[nodiscard]] std::uint64_t liveFullChunks() const
        {
            std::uint64_t live = 0u;
            for(auto const& entry : m_live)
                live += entry.second.full ? 1u : 0u;
            return live;
        }

        /** Highest number of live full-chunk ranges observed since construction. */
        [[nodiscard]] std::uint64_t peakLiveFullChunks() const
        {
            return m_peakLiveFull;
        }

        [[nodiscard]] std::uint64_t freeSlots() const
        {
            std::uint64_t slots = 0u;
            for(auto const& range : m_free)
                slots += range.length;
            slots += static_cast<std::uint64_t>(m_fullFree.size()) * m_fullChunkSize;
            return slots;
        }

        [[nodiscard]] std::uint64_t liveSlots() const
        {
            std::uint64_t slots = 0u;
            for(auto const& entry : m_live)
                slots += entry.second.length;
            return slots;
        }

        [[nodiscard]] std::uint32_t liveRangeCount() const
        {
            return static_cast<std::uint32_t>(m_live.size());
        }

        /** Longest free range of the general free list; zero when it is fully allocated.
         *
         * This is the widest range a general request could use, so it is the number that tells
         * whether a fixed-size general request can still be served. The full-chunk class is
         * reported by `freeFullChunks()` and the gap it can use by `fullGap()`.
         */
        [[nodiscard]] std::uint64_t largestFreeRange() const
        {
            std::uint64_t largest = 0u;
            for(auto const& range : m_free)
                largest = range.length > largest ? range.length : largest;
            return largest;
        }

        /** Slots between the general high-water mark and the full-chunk bump cursor. */
        [[nodiscard]] std::uint64_t fullGap() const
        {
            return m_fullCursor > m_generalHigh ? m_fullCursor - m_generalHigh : 0u;
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
            for(std::uint64_t const start : m_fullFree)
            {
                if(slot >= start && slot < start + m_fullChunkSize)
                    return true;
            }
            return false;
        }

        /** Free ranges of the general free list, ascending by start and coalesced. */
        [[nodiscard]] std::vector<Range> const& freeRanges() const
        {
            return m_free;
        }

        /** Released full-chunk ranges, ascending by start. */
        [[nodiscard]] std::vector<Range> fullFreeRanges() const
        {
            std::vector<Range> ranges;
            ranges.reserve(m_fullFree.size());
            for(std::uint64_t const start : m_fullFree)
                ranges.push_back(Range{start, m_fullChunkSize});
            return ranges;
        }

        /** Live ranges, ascending by start. */
        [[nodiscard]] std::vector<Range> liveRanges() const
        {
            std::vector<Range> live;
            live.reserve(m_live.size());
            for(auto const& entry : m_live)
                live.push_back(Range{entry.first, entry.second.length});
            return live;
        }

        /** Assert the arena invariants; throws `std::logic_error` on the first violation.
         *
         * The check walks the general free list, the released full-chunk ranges and the live
         * ranges together and proves three things: every range is non-empty and inside the arena,
         * every slot is covered exactly once (so no two live ranges overlap and no slot is owned
         * by two ranges), and the general free list is coalesced. Cost is `O(n log n)` in the
         * number of ranges, so it is a debug and test check, not part of the allocation path.
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
            std::vector<Range> const fullFree = fullFreeRanges();
            std::vector<Owned> ranges;
            ranges.reserve(m_free.size() + live.size() + fullFree.size());
            for(auto const& range : m_free)
                ranges.push_back(Owned{range.start, range.length, true});
            for(auto const& range : fullFree)
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
                // The general free list must stay coalesced. Released full-chunk ranges are kept
                // one per release, so two adjacent released blocks are legal there.
                if(i > 0u && range.free && ranges[i - 1u].free && range.start < m_fullCursor
                   && ranges[i - 1u].start + ranges[i - 1u].length == range.start)
                    throw std::logic_error{"graphFlow::SlotArena: adjacent general free ranges are not coalesced"};
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
            bool const generalInitial
                = m_free.size() == 1u && m_free.front().start == 0u && m_free.front().length == m_numSlots;
            if(!generalInitial || !m_fullFree.empty() || m_fullCursor != m_numSlots || m_generalHigh != 0u)
                throw std::logic_error{
                    "graphFlow::SlotArena: a full release must return the arena to its initial state"};
        }

    private:
        /** A live range plus whether it came from the full-chunk class. */
        struct Live
        {
            std::uint64_t length = 0u;
            bool full = false;
        };

        /** Take `count` slots from the top of the general free list and return their start. */
        std::uint64_t takeTopOfGeneral(std::uint64_t count)
        {
            if(m_free.empty())
                throw std::logic_error{"graphFlow::SlotArena: the full-chunk class needs a free general range"};
            Range& top = m_free.back();
            if(top.start + top.length != m_fullCursor)
                throw std::logic_error{
                    "graphFlow::SlotArena: the top general free range does not end at the full-chunk cursor"};
            if(top.length < count)
                throw std::logic_error{
                    "graphFlow::SlotArena: the top general free range is shorter than the full-chunk request"};
            std::uint64_t const start = top.start + top.length - count;
            if(start < m_generalHigh)
                throw std::logic_error{
                    "graphFlow::SlotArena: a full-chunk range would overlap the general high-water mark"};
            if(top.length == count)
                m_free.pop_back();
            else
                top.length -= count;
            m_fullCursor = start;
            return start;
        }

        /** Return a full-chunk range to the class so the next full-chunk request reuses it.
         *
         * A range at the bump cursor is adjacent to the general free list, so it merges back into
         * it and the cursor moves up; released ranges above it cascade. That is what makes a full
         * release restore the arena's initial state. Other ranges stay in the class free list.
         */
        void insertFullFree(std::uint64_t start)
        {
            if(start == m_fullCursor)
            {
                insertFree(Range{start, m_fullChunkSize});
                m_fullCursor += m_fullChunkSize;
                while(!m_fullFree.empty() && m_fullFree.front() == m_fullCursor)
                {
                    insertFree(Range{m_fullCursor, m_fullChunkSize});
                    m_fullCursor += m_fullChunkSize;
                    m_fullFree.erase(m_fullFree.begin());
                }
                return;
            }
            auto const at = std::lower_bound(m_fullFree.begin(), m_fullFree.end(), start);
            if(at != m_fullFree.end() && *at < start + m_fullChunkSize)
                throw std::logic_error{"graphFlow::SlotArena: a released full-chunk range overlaps a free one"};
            if(at != m_fullFree.begin() && *(at - 1) + m_fullChunkSize > start)
                throw std::logic_error{"graphFlow::SlotArena: a released full-chunk range overlaps a free one"};
            m_fullFree.insert(at, start);
        }

        /** Insert a returned general range and merge it with adjacent free neighbours. */
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
            // An arena with no live range has no high-water mark; resetting it keeps a full
            // release equal to the initial state and is safe because nothing owns a slot.
            if(m_free.size() == 1u && m_free.front().start == 0u && m_free.front().length == m_numSlots)
                m_generalHigh = 0u;
        }

        std::uint64_t m_numSlots;
        /** Slots of one full-chunk range; zero disables the class. */
        std::uint64_t m_fullChunkSize = 0u;
        /** Top of the free area the full-chunk class bumps down from. */
        std::uint64_t m_fullCursor = 0u;
        /** Highest slot ever handed out to a general request; the full class stays above it. */
        std::uint64_t m_generalHigh = 0u;
        std::vector<Range> m_free;
        /** Highest number of live full-chunk ranges observed. */
        std::uint64_t m_peakLiveFull = 0u;
        /** Released full-chunk starts, ascending. */
        std::vector<std::uint64_t> m_fullFree;
        std::map<std::uint64_t, Live> m_live;
    };

    /** Chunk-index to slot-range map of a slot-arena storage.
     *
     * The chunk-index free list, generations and reader leases stay in `detail::ChunkLedger`; this
     * map adds the range that each index owns. `acquire()` stores the start and length of the range
     * the `SlotArena` hands out, and `base(index)` returns the stored start instead of
     * `index * chunkCapacity`. A released index keeps `SlotArena::noSlot`, so reading the base of a
     * chunk that owns no range is a loud error rather than another chunk's memory.
     */
    class ArenaSlotMap
    {
    public:
        ArenaSlotMap(std::uint32_t numChunks, std::uint64_t numSlots, std::uint64_t fullChunkSize = 0u)
            : m_arena{numSlots, fullChunkSize}
            , m_bases(numChunks, SlotArena::noSlot)
            , m_fills(numChunks, 0u)
        {
            if(numChunks == 0u)
                throw std::invalid_argument{"graphFlow::ArenaSlotMap: numChunks must be non-zero"};
            // Every chunk owns at least one slot, so a map with more chunks than slots could
            // never satisfy an acquire.
            if(static_cast<std::uint64_t>(numChunks) > numSlots)
                throw std::invalid_argument{
                    "graphFlow::ArenaSlotMap: " + std::to_string(numChunks) + " chunks cannot be held by "
                    + std::to_string(numSlots) + " slots"};
        }

        /** Give `index` a contiguous range of exactly `fill` slots from the general region. */
        void acquire(std::uint32_t index, std::uint32_t fill)
        {
            checkIndex(index);
            if(fill == 0u)
                throw std::invalid_argument{"graphFlow::ArenaSlotMap: fill must be non-zero"};
            if(owns(index))
                throw std::logic_error{"graphFlow::ArenaSlotMap: chunk index already owns a slot range"};
            std::uint64_t const start = m_arena.allocate(fill);
            m_bases[index] = start;
            m_fills[index] = fill;
        }

        /** Give `index` one full-chunk range from the tail class.
         *
         * This is the output-chunk path: the range is served from the reserved region, so a free
         * list fragmented by small input ranges cannot block it. The committed fill is the full
         * chunk size, matching the range the scatter kernel addresses at `j % chunkCapacity`.
         */
        void acquireFull(std::uint32_t index)
        {
            checkIndex(index);
            if(owns(index))
                throw std::logic_error{"graphFlow::ArenaSlotMap: chunk index already owns a slot range"};
            if(m_arena.fullChunkSize() == 0u)
                throw std::logic_error{"graphFlow::ArenaSlotMap: the arena has no full-chunk size class"};
            std::uint64_t const start = m_arena.allocateFull();
            m_bases[index] = start;
            m_fills[index] = static_cast<std::uint32_t>(m_arena.fullChunkSize());
        }

        /** Return the range of `index` to the arena. */
        void release(std::uint32_t index)
        {
            checkIndex(index);
            if(!owns(index))
                throw std::logic_error{"graphFlow::ArenaSlotMap: chunk index owns no slot range"};
            m_arena.release(m_bases[index], m_fills[index]);
            m_bases[index] = SlotArena::noSlot;
            m_fills[index] = 0u;
        }

        [[nodiscard]] bool owns(std::uint32_t index) const
        {
            checkIndex(index);
            return m_bases[index] != SlotArena::noSlot;
        }

        /** Stored start slot of a chunk that owns a range. */
        [[nodiscard]] std::uint64_t base(std::uint32_t index) const
        {
            checkIndex(index);
            if(!owns(index))
                throw std::logic_error{"graphFlow::ArenaSlotMap: chunk index owns no slot range"};
            return m_bases[index];
        }

        /** Committed fill, i.e. the length of the owned range. */
        [[nodiscard]] std::uint32_t fill(std::uint32_t index) const
        {
            checkIndex(index);
            if(!owns(index))
                throw std::logic_error{"graphFlow::ArenaSlotMap: chunk index owns no slot range"};
            return m_fills[index];
        }

        [[nodiscard]] std::uint64_t slotCount() const
        {
            return m_arena.numSlots();
        }

        [[nodiscard]] std::uint64_t fullChunkSize() const
        {
            return m_arena.fullChunkSize();
        }

        [[nodiscard]] std::uint32_t liveRangeCount() const
        {
            return m_arena.liveRangeCount();
        }

        /** Highest number of live full-chunk ranges observed. */
        [[nodiscard]] std::uint64_t peakLiveFullChunks() const
        {
            return m_arena.peakLiveFullChunks();
        }

        /** Reserved full-chunk ranges that are not live. */
        [[nodiscard]] std::uint64_t freeFullChunks() const
        {
            return m_arena.freeFullChunks();
        }

        /** Reserved full-chunk ranges that are live. */
        [[nodiscard]] std::uint64_t liveFullChunks() const
        {
            return m_arena.liveFullChunks();
        }

        [[nodiscard]] SlotArena const& arena() const
        {
            return m_arena;
        }

        void verify() const
        {
            m_arena.verify();
        }

        void verifyReleased() const
        {
            m_arena.verifyReleased();
        }

    private:
        void checkIndex(std::uint32_t index) const
        {
            if(static_cast<std::size_t>(index) >= m_bases.size())
                throw std::invalid_argument{"graphFlow::ArenaSlotMap: chunk index out of range"};
        }

        SlotArena m_arena;
        std::vector<std::uint64_t> m_bases;
        std::vector<std::uint32_t> m_fills;
    };
} // namespace graphFlow
