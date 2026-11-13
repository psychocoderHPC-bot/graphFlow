/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/ChunkPool.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/ClassifyKernel.hpp"
#include "graphFlow/graph/DenseBinTable.hpp"
#include "graphFlow/graph/HashClassifyKernel.hpp"
#include "graphFlow/graph/HashScatterKernel.hpp"
#include "graphFlow/graph/Kernels.hpp"
#include "graphFlow/graph/ScatterKernel.hpp"
#include "graphFlow/rng/ParticleRng.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphFlow::particle
{
    /** Runtime knobs of the particle-feedback example. */
    struct Parameters
    {
        std::uint64_t numParticles = 0u;
        std::uint32_t chunkSize = 128u;
        std::uint32_t liveParticles = 0u; // P, zero means the documented default
        std::uint32_t stallTimeoutMs = 120'000u; // test hook: fail instead of spinning forever

        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 64u;
        std::uint32_t tileSize = 128u;
        std::uint32_t deviceIndex = 0u;
    };

    /** Checked number of tiles needed by a chunk.
     *
     * One thread block is launched per tile, so a zero fill launches no block and an empty chunk
     * never reaches a kernel. The division is performed in 64 bit because `fill + T - 1` can
     * overflow 32 bit when `fill` is close to `UINT32_MAX`; for `T >= 1` the result fits in
     * 32 bit because it never exceeds `fill`.
     */
    inline std::uint32_t tileCount(std::uint32_t fill, std::uint32_t tileSize)
    {
        if(tileSize == 0u)
            throw std::invalid_argument{"graphFlow: tile size must be non-zero"};
        if(fill == 0u)
            return 0u;
        std::uint64_t const tiles
            = (static_cast<std::uint64_t>(fill) + static_cast<std::uint64_t>(tileSize) - 1u) / tileSize;
        if(tiles > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error{"graphFlow: tile count exceeds the launchable block count"};
        return static_cast<std::uint32_t>(tiles);
    }

    struct HistogramRow
    {
        std::uint32_t weight = 0u;
        std::uint64_t count = 0u;

        friend bool operator==(HistogramRow const&, HistogramRow const&) = default;
    };

    struct FeedbackResult
    {
        std::vector<std::uint32_t> weights;   // final weight indexed by particle id
        std::vector<HistogramRow> histogram;  // sorted by weight, run-length encoded
        std::uint64_t cappedParticles = 0u;   // stopped by the --max-visits guard
        std::uint64_t sunkParticles = 0u;     // host-side sink conservation diagnostic
        std::uint32_t liveParticles = 0u;     // configured source-admission capacity P
        SchedulerStats stats;
    };

    /** Filter: `draw(seed, id, visit, Filter) >= property ? bin 1 : bin 0`. */
    struct FilterPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        std::uint64_t seed = 0u;

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            std::uint32_t visit) const
        {
            std::uint32_t const value = rng::draw(seed, id, visit, rng::purposeFilter);
            return value >= property ? 1u : 0u;
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            [[maybe_unused]] TAcc const& acc,
            [[maybe_unused]] std::uint32_t bin,
            [[maybe_unused]] Particle& particle,
            [[maybe_unused]] std::uint32_t* status,
            [[maybe_unused]] std::uint32_t* capped) const
        {
            // The filter preserves property, weight and visit.
        }
    };

    /** Shared implementation of the A and B feedback stages.
     *
     * A survives with `draw >= 2^31` and adds 1000; B survives with `draw >= 2^31` and adds 1.
     * Both increment `visit` only on feedback. A particle that would feed back but already reached
     * `maxVisits` is routed to the sink without a mutation and counted as non-terminating.
     */
    template<std::uint32_t T_purpose, std::uint32_t T_increment>
    struct ProcessPolicy
    {
        static constexpr std::uint32_t maxBins = 2u;

        std::uint64_t seed = 0u;
        std::uint32_t maxVisits = 0u;

        ALPAKA_FN_ACC bool survives(std::uint64_t id, std::uint32_t visit) const
        {
            return rng::draw(seed, id, visit, T_purpose) >= 0x80000000u;
        }

        ALPAKA_FN_ACC std::uint32_t classify(
            std::uint64_t id,
            [[maybe_unused]] std::uint32_t property,
            [[maybe_unused]] std::uint32_t weight,
            std::uint32_t visit) const
        {
            if(!survives(id, visit))
                return 0u; // sink
            return visit < maxVisits ? 1u : 0u; // 1 = feedback, 0 = sink due to the guard
        }

        template<typename TAcc>
        ALPAKA_FN_ACC void apply(
            TAcc const& acc,
            std::uint32_t bin,
            Particle& particle,
            std::uint32_t* status,
            std::uint32_t* capped) const
        {
            if(bin == 0u)
            {
                // `survives` is key-based and idempotent, so a stopped particle can be classified
                // again here without perturbing any draw position.
                if(survives(particle.id, particle.visit))
                    alpaka::onAcc::atomicAdd(acc, capped, std::uint32_t{1u}, alpaka::onAcc::scope::device);
                return;
            }
            if(particle.weight > std::numeric_limits<std::uint32_t>::max() - T_increment)
            {
                // Checked 32 bit addition: overflow is a hard batch error.
                alpaka::onAcc::atomicExch(acc, status, std::uint32_t{1u}, alpaka::onAcc::scope::device);
                return;
            }
            particle.weight += T_increment;
            particle.visit += 1u;
        }
    };

    using ProcessAPolicy = ProcessPolicy<rng::purposeProcessA, 1000u>;
    using ProcessBPolicy = ProcessPolicy<rng::purposeProcessB, 1u>;

    /** Source stage: enqueue the generate kernel for one admitted chunk. */
    template<typename TDevice>
    class GenerateStage
    {
    public:
        using Queue = typename GraphRunner<TDevice>::Queue;

        GenerateStage(ChunkPool<TDevice>& pool, std::uint64_t seed, std::uint32_t threadsPerBlock)
            : m_pool{pool}
            , m_seed{seed}
            , m_threads{threadsPerBlock}
        {
        }

        void launch(Queue const& queue, typename GraphRunner<TDevice>::Batch const& batch)
        {
            ChunkDescriptor const& chunk = batch.chunk;
            queue.enqueue(
                alpaka::onHost::ThreadSpec{std::uint32_t{1u}, m_threads},
                alpaka::KernelBundle{
                    GenerateKernel{m_seed},
                    m_pool.ids(),
                    m_pool.properties(),
                    m_pool.weights(),
                    m_pool.visits(),
                    batch.firstId,
                    static_cast<std::uint32_t>(m_pool.base(chunk.handle)),
                    chunk.fill});
        }

        bool advance(Queue const&, ChunkDescriptor const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        }

        void finish(ChunkDescriptor const& chunk, std::vector<ChunkDescriptor>& outputs) const
        {
            outputs.push_back(chunk);
        }

    private:
        ChunkPool<TDevice>& m_pool;
        std::uint64_t m_seed;
        std::uint32_t m_threads;
    };

    /** Transform stage: classify, plan output chunks on the host, scatter. */
    template<typename TDevice, typename TPolicy, typename TBinTable = DenseBinTable>
    class TransformStage
    {
    public:
        using Queue = typename GraphRunner<TDevice>::Queue;
        using U32View
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedU32View
            = ALPAKA_TYPEOF(alpaka::onHost::allocMapped<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

        /** `tileSize` is the logical tile width T, not the resolved block extent.
         *
         * The stage launches one block per tile with
         * `ThreadSpec{tileCount(fill, T), GraphRunner::threadsPerBlock(device, T)}`. On the host
         * backend the block extent collapses to one real thread while T keeps the tile slice
         * width used by the kernels, so a chunk larger than T still splits into several blocks.
         */
        TransformStage(
            TDevice const& device,
            ChunkPool<TDevice>& pool,
            TPolicy policy,
            U32View capped,
            std::uint32_t tileSize)
            : m_pool{pool}
            , m_policy{policy}
            , m_capacity{pool.chunkCapacity()}
            , m_bins{TPolicy::maxBins}
            , m_reserveBound{static_cast<std::uint32_t>(
                  outputChunkBound(pool.chunkCapacity(), pool.chunkCapacity(), TPolicy::maxBins))}
            , m_tileSize{tileSize}
            , m_threads{GraphRunner<TDevice>::threadsPerBlock(device, tileSize)}
            , m_counts{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins) + 1u})}
            , m_countsHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins) + 1u})}
            , m_next{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins)})}
            , m_nextHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins)})}
            , m_table{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins) + m_reserveBound})}
            , m_tableHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(TPolicy::maxBins) + m_reserveBound})}
            , m_capped{std::move(capped)}
        {
            m_totals.assign(m_bins, 0u);
            m_pending.reserve(m_reserveBound);
        }

        void launch(Queue const& queue, typename GraphRunner<TDevice>::Batch const& batch)
        {
            throwIfFailed();
            ChunkDescriptor const& chunk = batch.chunk;
            m_stage = 0u;
            m_pending.clear();

            // total[B] + status word; both start at zero for this batch.
            alpaka::onHost::memset(queue, m_counts, std::uint8_t{0});
            std::uint32_t const tiles = tileCount(chunk.fill, m_tileSize);
            if(tiles != 0u)
            {
                if constexpr(TBinTable::isHash)
                {
                    queue.enqueue(
                        alpaka::onHost::ThreadSpec{tiles, m_threads},
                        alpaka::KernelBundle{
                            HashClassifyKernel<TPolicy, TBinTable>{m_policy},
                            m_pool.ids(),
                            m_pool.properties(),
                            m_pool.weights(),
                            m_pool.visits(),
                            m_pool.binIds(),
                            m_counts,
                            chunk.handle.index,
                            m_capacity,
                            chunk.fill,
                            m_tileSize,
                            m_bins});
                }
                else
                {
                    queue.enqueue(
                        alpaka::onHost::ThreadSpec{tiles, m_threads},
                        alpaka::KernelBundle{
                            ClassifyKernel<TPolicy>{m_policy},
                            m_pool.ids(),
                            m_pool.properties(),
                            m_pool.weights(),
                            m_pool.visits(),
                            m_pool.binIds(),
                            m_counts,
                            chunk.handle.index,
                            m_capacity,
                            chunk.fill,
                            m_tileSize});
                }
            }
            // The event the scheduler arms after this call covers the metadata copy.
            alpaka::onHost::memcpy(queue, m_countsHost, m_counts);
        }

        bool advance(Queue const& queue, ChunkDescriptor const& chunk, std::uint32_t& allocated)
        {
            throwIfFailed();
            allocated = 0u;
            if(m_stage == 0u)
            {
                planAndScatter(queue, chunk, allocated);
                m_stage = 1u;
                return true;
            }
            try
            {
                finalise();
            }
            catch(...)
            {
                // The scatter completed but its result is unusable; hand the acquired output chunks
                // back before the runner tears the batch down.
                releasePending();
                resetBatchState();
                m_failed = true;
                throw;
            }
            m_stage = 0u;
            return false;
        }

        void finish(ChunkDescriptor const&, std::vector<ChunkDescriptor>& outputs) const
        {
            outputs.insert(outputs.end(), m_pending.begin(), m_pending.end());
        }

        /** Conservative number of output chunks this stage may need; used by tests/docs. */
        [[nodiscard]] std::uint32_t reserveBound() const
        {
            return m_reserveBound;
        }

    private:
        void planAndScatter(Queue const& queue, ChunkDescriptor const& chunk, std::uint32_t& allocated)
        {
            std::uint32_t const tiles = tileCount(chunk.fill, m_tileSize);
            std::uint32_t const* host = m_countsHost.data();
            if(host[m_bins] != 0u)
                throw BatchError{"classify: device status reports an invalid classifier result"};

            for(std::uint32_t bin = 0u; bin < m_bins; ++bin)
                m_totals[bin] = host[bin];

            DenseBinPlan const plan = planDenseBins(m_totals, m_capacity);
            if(plan.numOutputChunks > m_reserveBound)
                throw BatchError{"plan: exact chunk count exceeds the reservation bound"};

            std::vector<std::uint32_t> chunkIndices;
            chunkIndices.reserve(plan.numOutputChunks);
            m_pending.clear();
            try
            {
                for(std::size_t bin = 0u; bin < m_totals.size(); ++bin)
                {
                    std::uint32_t const chunks
                        = static_cast<std::uint32_t>(ceilDivChecked(m_totals[bin], m_capacity));
                    for(std::uint32_t k = 0u; k < chunks; ++k)
                    {
                        ChunkHandle const handle = m_pool.acquire();
                        chunkIndices.push_back(handle.index);
                        std::uint32_t const fillIndex = plan.binBase[bin] + k;
                        m_pending.push_back(
                            ChunkDescriptor{handle, plan.fills[fillIndex], static_cast<std::uint32_t>(bin)});
                    }
                }

                std::vector<std::uint32_t> const table = buildDenseTable(plan, chunkIndices);
                if(static_cast<std::uint64_t>(table.size()) > m_table.getExtents().x())
                    throw BatchError{"plan: device table buffer too small"};
                std::copy(table.begin(), table.end(), m_tableHost.data());
                alpaka::onHost::memcpy(queue, m_table, m_tableHost);
                alpaka::onHost::memset(queue, m_next, std::uint8_t{0});

                if constexpr(TBinTable::isHash)
                {
                    if(tiles != 0u)
                    {
                        queue.enqueue(
                            alpaka::onHost::ThreadSpec{tiles, m_threads},
                            alpaka::KernelBundle{
                                HashScatterKernel<TPolicy, TBinTable>{m_policy},
                                m_pool.ids(),
                                m_pool.properties(),
                                m_pool.weights(),
                                m_pool.visits(),
                                m_pool.binIds(),
                                m_pool.ids(),
                                m_pool.properties(),
                                m_pool.weights(),
                                m_pool.visits(),
                                m_table,
                                m_next,
                                m_counts,
                                m_capped,
                                chunk.handle.index,
                                m_capacity,
                                chunk.fill,
                                m_tileSize,
                                m_bins});
                    }
                }
                else
                {
                    if(tiles != 0u)
                    {
                        queue.enqueue(
                            alpaka::onHost::ThreadSpec{tiles, m_threads},
                            alpaka::KernelBundle{
                                ScatterKernel<TPolicy>{m_policy},
                                m_pool.ids(),
                                m_pool.properties(),
                                m_pool.weights(),
                                m_pool.visits(),
                                m_pool.binIds(),
                                m_pool.ids(),
                                m_pool.properties(),
                                m_pool.weights(),
                                m_pool.visits(),
                                m_table,
                                m_next,
                                m_counts,
                                m_capped,
                                chunk.handle.index,
                                m_capacity,
                                chunk.fill,
                                m_tileSize});
                    }
                }

                // The completion event must cover the status word and the range counters.
                alpaka::onHost::memcpy(queue, m_countsHost, m_counts);
                alpaka::onHost::memcpy(queue, m_nextHost, m_next);
            }
            catch(...)
            {
                // Release once here; the runner separately releases the input chunk. The batch
                // state is reset as well, so a failed plan cannot leave stale totals or host
                // counters behind, and the stage is marked single-use because the device stage may
                // be partially queued.
                releasePending();
                resetBatchState();
                m_failed = true;
                throw;
            }
            allocated = static_cast<std::uint32_t>(chunkIndices.size());
        }

        /** Clear every per-batch field so a failure cannot leak partially planned state. */
        void resetBatchState()
        {
            m_stage = 0u;
            std::fill(m_totals.begin(), m_totals.end(), 0u);
            std::fill_n(m_countsHost.data(), static_cast<std::size_t>(m_bins) + 1u, 0u);
        }

        /** A failed batch may have left a partially queued device stage, so the instance must not
         * be reused; callers construct a fresh stage for the next run. */
        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{
                    "graphFlow: TransformStage is single-use after a failed batch"};
        }

        void releasePending()
        {
            for(ChunkDescriptor const& descriptor : m_pending)
                m_pool.release(descriptor.handle);
            m_pending.clear();
        }

        void finalise()
        {
            if(m_countsHost.data()[m_bins] != 0u)
                throw BatchError{"scatter: device status reports a checked weight overflow"};
            for(std::uint32_t bin = 0u; bin < m_bins; ++bin)
            {
                if(m_nextHost.data()[bin] != m_totals[bin])
                    throw BatchError{"scatter: next[bin] does not match total[bin]"};
            }
        }

        ChunkPool<TDevice>& m_pool;
        TPolicy m_policy;
        std::uint32_t m_capacity;
        std::uint32_t m_bins;
        std::uint32_t m_reserveBound;
        std::uint32_t m_tileSize;
        std::uint32_t m_threads;
        U32View m_counts;
        MappedU32View m_countsHost;
        U32View m_next;
        MappedU32View m_nextHost;
        U32View m_table;
        MappedU32View m_tableHost;
        U32View m_capped;
        std::uint32_t m_stage = 0u;
        std::vector<std::uint32_t> m_totals;
        std::vector<ChunkDescriptor> m_pending;
        bool m_failed = false;
    };

    /** Sink stage: write `result[id] = weight` for one chunk. */
    template<typename TDevice>
    class SinkStage
    {
    public:
        using Queue = typename GraphRunner<TDevice>::Queue;
        using U32View
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

        SinkStage(ChunkPool<TDevice>& pool, U32View result, std::uint32_t threadsPerBlock)
            : m_pool{pool}
            , m_result{std::move(result)}
            , m_threads{threadsPerBlock}
        {
        }

        void launch(Queue const& queue, typename GraphRunner<TDevice>::Batch const& batch)
        {
            ChunkDescriptor const& chunk = batch.chunk;
            m_sunk += chunk.fill;
            queue.enqueue(
                alpaka::onHost::ThreadSpec{std::uint32_t{1u}, m_threads},
                alpaka::KernelBundle{
                    SinkKernel{},
                    m_result,
                    m_pool.ids(),
                    m_pool.weights(),
                    static_cast<std::uint32_t>(m_pool.base(chunk.handle)),
                    chunk.fill});
        }

        bool advance(Queue const&, ChunkDescriptor const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        }

        void finish(ChunkDescriptor const&, std::vector<ChunkDescriptor>&) const
        {
        }

        [[nodiscard]] std::uint64_t sunkParticles() const
        {
            return m_sunk;
        }

    private:
        ChunkPool<TDevice>& m_pool;
        U32View m_result;
        std::uint32_t m_threads;
        std::uint64_t m_sunk = 0u;
    };

    /** Build the sorted, run-length encoded weight histogram. */
    inline std::vector<HistogramRow> histogramOf(std::vector<std::uint32_t> weights)
    {
        std::sort(weights.begin(), weights.end());
        std::vector<HistogramRow> rows;
        for(std::size_t i = 0u; i < weights.size();)
        {
            std::size_t j = i;
            while(j < weights.size() && weights[j] == weights[i])
                ++j;
            rows.push_back(HistogramRow{weights[i], static_cast<std::uint64_t>(j - i)});
            i = j;
        }
        return rows;
    }
} // namespace graphFlow::particle
