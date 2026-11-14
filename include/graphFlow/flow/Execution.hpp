/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Chunk.hpp"
#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/LaunchPolicy.hpp"
#include "graphFlow/core/PayloadSlotStorage.hpp"
#include "graphFlow/core/Reservation.hpp"
#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/PayloadStorage.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/graph/BinTable.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/graph/DenseBinTable.hpp"
#include "graphFlow/graph/TileDescriptor.hpp"
#include "graphFlow/sched/GraphRunner.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace graphFlow::detail
{
    /** Whether `TFunctor` accepts the real accelerator as its first argument. */
    template<typename TFunctor, typename TAcc, typename... TArgs>
    concept InvocableWithAccelerator = requires(TFunctor const& functor, TAcc const& accelerator, TArgs&&... args) {
        functor(accelerator, std::forward<TArgs>(args)...);
    };

    /** Whether `TFunctor` accepts the accelerator followed by a status reference. */
    template<typename TFunctor, typename TAcc, typename TStatus, typename... TArgs>
    concept InvocableWithAcceleratorAndStatus
        = requires(TFunctor const& functor, TAcc const& accelerator, TStatus& status, TArgs&&... args) {
              functor(accelerator, status, std::forward<TArgs>(args)...);
          };

    /** Whether `TFunctor` accepts a status reference without an accelerator. */
    template<typename TFunctor, typename TStatus, typename... TArgs>
    concept InvocableWithStatus = requires(TFunctor const& functor, TStatus& status, TArgs&&... args) {
        functor(status, std::forward<TArgs>(args)...);
    };

    /** Engine-aware invocation shapes. The engine follows the optional accelerator and precedes
     * the optional status, matching the documented `operator()(acc, Engine&, Status&, Payload&)`. */
    template<typename TFunctor, typename TAcc, typename TEngine, typename TStatus, typename... TArgs>
    concept InvocableWithAcceleratorEngineAndStatus = requires(
        TFunctor const& functor,
        TAcc const& accelerator,
        TEngine& engine,
        TStatus& status,
        TArgs&&... args) { functor(accelerator, engine, status, std::forward<TArgs>(args)...); };

    template<typename TFunctor, typename TAcc, typename TEngine, typename... TArgs>
    concept InvocableWithAcceleratorAndEngine
        = requires(TFunctor const& functor, TAcc const& accelerator, TEngine& engine, TArgs&&... args) {
              functor(accelerator, engine, std::forward<TArgs>(args)...);
          };

    /** Any supported invocation shape for the given argument tail. */
    template<typename TFunctor, typename TAcc, typename TEngine, typename TStatus, typename... TArgs>
    concept ContextInvocable
        = InvocableWithAcceleratorEngineAndStatus<TFunctor, TAcc, TEngine, TStatus, TArgs...>
          || InvocableWithAcceleratorAndEngine<TFunctor, TAcc, TEngine, TArgs...>
          || InvocableWithAcceleratorAndStatus<TFunctor, TAcc, TStatus, TArgs...>
          || InvocableWithAccelerator<TFunctor, TAcc, TArgs...> || InvocableWithStatus<TFunctor, TStatus, TArgs...>
          || std::invocable<TFunctor const&, TArgs...>;

    /** Invoke a public functor with the optional accelerator, engine and status arguments.
     *
     * Engine forms take priority so an opting-in functor receives its initialized stream. Existing
     * accelerator and status forms keep their meaning through the fallback order below.
     */
    template<typename TFunctor, typename TAcc, typename TEngine, typename TStatus, typename... TArgs>
    ALPAKA_FN_ACC decltype(auto) invokeWithContext(
        TFunctor const& functor,
        TAcc const& accelerator,
        TEngine& engine,
        TStatus& status,
        TArgs&&... args)
    {
        if constexpr(InvocableWithAcceleratorEngineAndStatus<TFunctor, TAcc, TEngine, TStatus, TArgs...>)
            return functor(accelerator, engine, status, std::forward<TArgs>(args)...);
        else if constexpr(InvocableWithAcceleratorAndEngine<TFunctor, TAcc, TEngine, TArgs...>)
            return functor(accelerator, engine, std::forward<TArgs>(args)...);
        else if constexpr(InvocableWithAcceleratorAndStatus<TFunctor, TAcc, TStatus, TArgs...>)
            return functor(accelerator, status, std::forward<TArgs>(args)...);
        else if constexpr(InvocableWithAccelerator<TFunctor, TAcc, TArgs...>)
            return functor(accelerator, std::forward<TArgs>(args)...);
        else if constexpr(InvocableWithStatus<TFunctor, TStatus, TArgs...>)
            return functor(status, std::forward<TArgs>(args)...);
        else
            return functor(std::forward<TArgs>(args)...);
    }

    /** Full tiles of one chunk; zero fill needs no launch. */
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

    /** Apply a transform functor, accepting a mutable payload or a returned replacement. */
    template<typename TPayload, typename TInvoke>
    ALPAKA_FN_ACC TPayload finishTransform(TPayload& payload, TInvoke&& invoke)
    {
        using Result = std::invoke_result_t<TInvoke>;
        if constexpr(std::is_void_v<Result>)
        {
            std::forward<TInvoke>(invoke)();
        }
        else
        {
            payload = static_cast<TPayload>(std::forward<TInvoke>(invoke)());
        }
        return payload;
    }

    template<typename TFunctor, typename TAcc, typename TEngine, typename TStatus, typename TPayload>
    ALPAKA_FN_ACC TPayload applyTransform(
        TFunctor const& functor,
        TAcc const& accelerator,
        TEngine& engine,
        TStatus& status,
        TPayload payload)
    {
        if constexpr(ContextInvocable<TFunctor, TAcc, TEngine, TStatus, TPayload&>)
        {
            auto invoke
                = [&]() -> decltype(auto) { return invokeWithContext(functor, accelerator, engine, status, payload); };
            return finishTransform(payload, invoke);
        }
        else
        {
            auto invoke = [&]() -> decltype(auto)
            { return invokeWithContext(functor, accelerator, engine, status, std::as_const(payload)); };
            return finishTransform(payload, invoke);
        }
    }

    /** Apply an optional partition updater and normalise its result. */
    template<typename TInvoke>
    ALPAKA_FN_ACC UpdateResult finishUpdate(TInvoke&& invoke)
    {
        using Result = std::invoke_result_t<TInvoke>;
        if constexpr(std::is_void_v<Result>)
        {
            std::forward<TInvoke>(invoke)();
            return UpdateResult::success;
        }
        else
        {
            return static_cast<UpdateResult>(std::forward<TInvoke>(invoke)());
        }
    }

    template<typename TUpdate, typename TAcc, typename TEngine, typename TStatus, typename TPayload, typename TBin>
    ALPAKA_FN_ACC UpdateResult applyUpdate(
        TUpdate const& update,
        TAcc const& accelerator,
        TEngine& engine,
        TStatus& status,
        TBin bin,
        TPayload& payload)
    {
        auto invoke
            = [&]() -> decltype(auto) { return invokeWithContext(update, accelerator, engine, status, bin, payload); };
        return finishUpdate(invoke);
    }

    /** Load/store trait of a payload storage.
     *
     * The trait exposes four independent payload views. The array-of-struct default binds all four
     * to the same contiguous payload array; the particle SoA specialization binds them to ids,
     * properties, weights and visits. Keeping the views as separate kernel arguments is required
     * because alpaka only rewrites top-level view arguments for the device.
     *
     * `load` and `store` are templates so they accept both the host view type and the
     * device-accessible view type that `KernelBundle` derives from it.
     */
    template<typename TStorage, typename TPayload>
    struct StorageAccess
    {
        using Payload = TPayload;
        using View0 = std::remove_cvref_t<decltype(std::declval<TStorage const&>().payload())>;
        using View1 = View0;
        using View2 = View0;
        using View3 = View0;

        [[nodiscard]] static View0 view0(TStorage const& storage)
        {
            return storage.payload();
        }

        [[nodiscard]] static View1 view1(TStorage const& storage)
        {
            return storage.payload();
        }

        [[nodiscard]] static View2 view2(TStorage const& storage)
        {
            return storage.payload();
        }

        [[nodiscard]] static View3 view3(TStorage const& storage)
        {
            return storage.payload();
        }

        template<typename V0, typename V1, typename V2, typename V3>
        ALPAKA_FN_ACC Payload load(V0 const& view0, V1 const&, V2 const&, V3 const&, std::uint32_t slot) const
        {
            return view0[slot];
        }

        template<typename V0, typename V1, typename V2, typename V3>
        ALPAKA_FN_ACC void store(V0& view0, V1&, V2&, V3&, std::uint32_t slot, Payload const& payload) const
        {
            view0[slot] = payload;
        }
    };

    /** Particle SoA access: the payload is assembled from four field views. */
    template<typename TDevice>
    struct StorageAccess<ParticleStorage<TDevice>, Particle>
    {
        using Payload = Particle;
        using View0 = typename ParticleStorage<TDevice>::IdView;
        using View1 = typename ParticleStorage<TDevice>::U32View;
        using View2 = typename ParticleStorage<TDevice>::U32View;
        using View3 = typename ParticleStorage<TDevice>::U32View;

        [[nodiscard]] static View0 view0(ParticleStorage<TDevice> const& storage)
        {
            return storage.ids();
        }

        [[nodiscard]] static View1 view1(ParticleStorage<TDevice> const& storage)
        {
            return storage.properties();
        }

        [[nodiscard]] static View2 view2(ParticleStorage<TDevice> const& storage)
        {
            return storage.weights();
        }

        [[nodiscard]] static View3 view3(ParticleStorage<TDevice> const& storage)
        {
            return storage.visits();
        }

        template<typename VId, typename VProperty, typename VWeight, typename VVisit>
        ALPAKA_FN_ACC Payload load(
            VId const& ids,
            VProperty const& properties,
            VWeight const& weights,
            VVisit const& visits,
            std::uint32_t slot) const
        {
            return Payload{ids[slot], properties[slot], weights[slot], visits[slot]};
        }

        template<typename VId, typename VProperty, typename VWeight, typename VVisit>
        ALPAKA_FN_ACC void store(
            VId& ids,
            VProperty& properties,
            VWeight& weights,
            VVisit& visits,
            std::uint32_t slot,
            Payload const& payload) const
        {
            ids[slot] = payload.id;
            properties[slot] = payload.property;
            weights[slot] = payload.weight;
            visits[slot] = payload.visit;
        }
    };

    /** Source kernel: one launch materialises one admitted chunk. */
    template<typename TFunctor, typename TAccess>
    struct SourceKernel
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto view0,
            auto view1,
            auto view2,
            auto view3,
            alpaka::concepts::IMdSpan auto statusView,
            std::uint64_t firstId,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const offset = static_cast<std::uint32_t>(i);
                std::uint64_t const id = firstId + static_cast<std::uint64_t>(offset);
                rng::Engine engine = rng::makeEngine(seed, id, 0u, purpose);
                auto payload = invokeWithContext(functor, acc, engine, status, id);
                access
                    .store(view0, view1, view2, view3, base + offset, static_cast<typename TAccess::Payload>(payload));
            }
        }
    };

    /** Out-of-place transform kernel: one launch copies the whole batch.
     *
     * Each block owns one input chunk descriptor and threads stride over its payloads. The output
     * position is the dense concatenation of the batch, so `R = ceil(sum(fill) / C)` output chunks
     * hold the result and no histogram or scatter is involved.
     */
    template<typename TFunctor, typename TAccess>
    struct TransformKernel
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto in0,
            auto in1,
            auto in2,
            auto in3,
            auto out0,
            auto out1,
            auto out2,
            auto out3,
            alpaka::concepts::IMdSpan auto statusView,
            alpaka::concepts::IMdSpan auto inBase,
            alpaka::concepts::IMdSpan auto inCount,
            alpaka::concepts::IMdSpan auto outPosition,
            alpaka::concepts::IMdSpan auto outChunks,
            std::uint32_t chunkCapacity,
            std::uint32_t chunks) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};
            for(auto [chunkIndex] : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{chunks}))
            {
                std::uint32_t const chunk = static_cast<std::uint32_t>(chunkIndex);
                std::uint32_t const inSlot = inBase[chunk];
                std::uint32_t const count = inCount[chunk];
                std::uint32_t const position = outPosition[chunk];
                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const offset = static_cast<std::uint32_t>(i);
                    std::uint32_t const global = position + offset;
                    std::uint32_t const outputChunk = global / chunkCapacity;
                    std::uint32_t const outputOffset = global % chunkCapacity;
                    std::uint32_t const outSlot = outChunks[outputChunk] * chunkCapacity + outputOffset;
                    typename TAccess::Payload payload = access.load(in0, in1, in2, in3, inSlot + offset);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    payload = applyTransform(functor, acc, engine, status, std::move(payload));
                    access.store(out0, out1, out2, out3, outSlot, payload);
                }
            }
        }
    };

    /** Dense classify kernel of one typed partition batch. */
    template<typename TClassifier, typename TAccess, std::uint32_t T_B = TClassifier::maxBins>
    struct PartitionClassifyKernel
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");

        TClassifier classifier;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto view0,
            auto view1,
            auto view2,
            auto view3,
            alpaka::concepts::IMdSpan auto statusView,
            alpaka::concepts::IMdSpan auto binIds,
            alpaka::concepts::IMdSpan auto counts,
            alpaka::concepts::IMdSpan auto tiles,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};

            std::uint32_t const numTiles = static_cast<std::uint32_t>(tiles.getExtents().x());
            auto tileCounts = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});

            for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
            {
                std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                TileDescriptor const tile = tiles[tileIndex];
                std::uint32_t const base = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                std::uint32_t const count = tile.count;

                for(auto [bin] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                    tileCounts[bin] = 0u;
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const slot = base + static_cast<std::uint32_t>(i);
                    typename TAccess::Payload const payload = access.load(view0, view1, view2, view3, slot);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    std::uint32_t const bin
                        = static_cast<std::uint32_t>(invokeWithContext(classifier, acc, engine, status, payload));
                    if(bin >= T_B)
                    {
                        status.set(StatusCode::invalid);
                        counts[T_B] = 1u;
                        continue;
                    }
                    binIds[slot] = bin;
                    onAcc::atomicAdd(acc, &tileCounts[bin], std::uint32_t{1u}, onAcc::scope::block);
                }
                onAcc::syncBlockThreads(acc);

                for(auto [bin] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                {
                    std::uint32_t const inTile = tileCounts[bin];
                    if(inTile != 0u)
                        onAcc::atomicAdd(acc, &counts[bin], inTile, onAcc::scope::device);
                }
            }
        }
    };

    /** Dense scatter kernel of one typed partition batch.
     *
     * The selected bin is read from `binIds`, the updater runs once per emitted payload, and the
     * mutated payload is written to the planned output slot. `counts[T_B]` is the internal status
     * word; a checked updater overflow reports it and leaves the payload unchanged in the output.
     */
    template<
        typename TClassifier,
        typename TUpdate,
        typename TAccess,
        typename TBin = ClassifierBin<TClassifier, typename TAccess::Payload>,
        std::uint32_t T_B = TClassifier::maxBins>
    struct PartitionScatterKernel
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");

        TUpdate update;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto in0,
            auto in1,
            auto in2,
            auto in3,
            auto out0,
            auto out1,
            auto out2,
            auto out3,
            alpaka::concepts::IMdSpan auto statusView,
            alpaka::concepts::IMdSpan auto binIds,
            alpaka::concepts::IMdSpan auto table,
            alpaka::concepts::IMdSpan auto next,
            alpaka::concepts::IMdSpan auto counts,
            alpaka::concepts::IMdSpan auto tiles,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};

            std::uint32_t const numTiles = static_cast<std::uint32_t>(tiles.getExtents().x());
            auto tileCounts = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
            auto tileBase = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
            auto tileCursor = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});

            for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
            {
                std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                TileDescriptor const tile = tiles[tileIndex];
                std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                std::uint32_t const count = tile.count;

                for(auto [bin] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                    tileCounts[bin] = 0u;
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const bin = binIds[inBase + static_cast<std::uint32_t>(i)];
                    onAcc::atomicAdd(acc, &tileCounts[bin], std::uint32_t{1u}, onAcc::scope::block);
                }
                onAcc::syncBlockThreads(acc);

                for(auto [bin] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                {
                    std::uint32_t const inTile = tileCounts[bin];
                    if(inTile != 0u)
                    {
                        tileBase[bin] = onAcc::atomicAdd(acc, &next[bin], inTile, onAcc::scope::device);
                        tileCursor[bin] = 0u;
                    }
                }
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const slotIn = inBase + static_cast<std::uint32_t>(i);
                    std::uint32_t const bin = binIds[slotIn];
                    std::uint32_t const rank
                        = onAcc::atomicAdd(acc, &tileCursor[bin], std::uint32_t{1u}, onAcc::scope::block);
                    std::uint32_t const j = tileBase[bin] + rank;
                    std::uint32_t const outputChunk = table[std::uint32_t{T_B} + table[bin] + j / chunkCapacity];
                    std::uint32_t const outputOffset = j % chunkCapacity;

                    typename TAccess::Payload payload = access.load(in0, in1, in2, in3, slotIn);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    UpdateResult const result
                        = applyUpdate(update, acc, engine, status, static_cast<TBin>(bin), payload);
                    if(result == UpdateResult::invalid)
                        status.set(StatusCode::invalid);
                    else if(result == UpdateResult::overflow)
                        status.set(StatusCode::overflow);
                    if(result != UpdateResult::success)
                        onAcc::atomicExch(acc, &counts[T_B], std::uint32_t{1u}, onAcc::scope::device);

                    access.store(out0, out1, out2, out3, outputChunk * chunkCapacity + outputOffset, payload);
                }
            }
        }
    };

    /** Sink kernel: consume one chunk without producing an output descriptor. */
    template<typename TFunctor, typename TAccess>
    struct SinkKernel
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto view0,
            auto view1,
            auto view2,
            auto view3,
            alpaka::concepts::IMdSpan auto statusView,
            std::uint32_t base,
            std::uint32_t fill) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint32_t const offset = static_cast<std::uint32_t>(i);
                typename TAccess::Payload const payload = access.load(view0, view1, view2, view3, base + offset);
                rng::Engine engine = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                invokeWithContext(functor, acc, engine, status, payload);
            }
        }
    };

    /** Type-erased host adapter the scheduler hooks call. */
    template<typename TDevice, typename TStorage>
    class StageBase
    {
    public:
        using Runner = GraphRunner<TDevice, TStorage>;
        using Queue = typename Runner::Queue;
        using Batch = typename Runner::Batch;

        virtual ~StageBase() = default;

        virtual void launch(Queue const& queue, Batch const& batch) = 0;
        virtual bool advance(Queue const& queue, Batch const& batch, std::uint32_t& allocated) = 0;
        virtual void finish(Batch const& batch, std::vector<ChunkDescriptor>& outputs) = 0;
        virtual void abort() = 0;
    };

    /** Build the scheduler hooks of one typed stage. The stage outlives the runner. */
    template<typename TDevice, typename TStorage>
    typename GraphRunner<TDevice, TStorage>::Hooks makeHooks(StageBase<TDevice, TStorage>* stage)
    {
        using Runner = GraphRunner<TDevice, TStorage>;
        typename Runner::Hooks hooks;
        hooks.launch = [stage](typename Runner::Queue const& queue, typename Runner::Batch const& batch)
        { stage->launch(queue, batch); };
        hooks.advance =
            [stage](typename Runner::Queue const& queue, typename Runner::Batch const& batch, std::uint32_t& allocated)
        { return stage->advance(queue, batch, allocated); };
        hooks.finish = [stage](typename Runner::Batch const& batch, std::vector<ChunkDescriptor>& outputs)
        { stage->finish(batch, outputs); };
        hooks.abort = [stage]() { stage->abort(); };
        return hooks;
    }

    /** Source stage: one launch per admitted chunk, one output descriptor. */
    template<typename TDevice, typename TStorage, typename TPayload, typename TFunctor>
    class SourceStage final : public StageBase<TDevice, TStorage>
    {
        using Base = StageBase<TDevice, TStorage>;
        using Access = StorageAccess<TStorage, TPayload>;
        using U32View = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedU32View = ALPAKA_TYPEOF(
            alpaka::onHost::allocMapped<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

    public:
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        SourceStage(
            TDevice const& device,
            TStorage& storage,
            Config const& config,
            TFunctor functor,
            std::uint32_t purpose = 0u)
            : m_device{device}
            , m_storage{storage}
            , m_functor{std::move(functor)}
            , m_seed{config.seed}
            , m_purpose{purpose}
            , m_tileSize{config.tileSize}
            , m_frameCap{config.frameCap}
            , m_status{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
            , m_statusHost{alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
        {
            if(m_frameCap == 0u)
                throw std::invalid_argument{"graphFlow: frameCap must be non-zero"};
        }

        void launch(Queue const& queue, Batch const& batch) override
        {
            throwIfFailed();
            ChunkDescriptor const& chunk = batch.chunk();
            // Reset the single status word on the node queue, then copy it back only after the
            // source kernel, so `advance` observes a status that covers this batch.
            alpaka::onHost::memset(queue, m_status, std::uint8_t{0});
            if(chunk.fill != 0u)
            {
                ++m_launches;
                LaunchPolicy const policy = launchPolicy(m_device, chunk.fill, m_tileSize, m_frameCap);
                queue.enqueue(
                    policy.frameSpec(),
                    alpaka::KernelBundle{
                        SourceKernel<TFunctor, Access>{m_functor, m_seed, m_purpose},
                        Access::view0(m_storage),
                        Access::view1(m_storage),
                        Access::view2(m_storage),
                        Access::view3(m_storage),
                        m_status,
                        batch.firstId,
                        static_cast<std::uint32_t>(m_storage.base(chunk.handle)),
                        chunk.fill});
            }
            alpaka::onHost::memcpy(queue, m_statusHost, m_status);
        }

        bool advance(Queue const&, Batch const&, std::uint32_t& allocated) override
        {
            throwIfFailed();
            checkStatus("source");
            allocated = 0u;
            return false;
        }

        void finish(Batch const& batch, std::vector<ChunkDescriptor>& outputs) override
        {
            outputs.insert(outputs.end(), batch.chunks.begin(), batch.chunks.end());
        }

        void abort() override
        {
            m_failed = true;
        }

        [[nodiscard]] std::uint32_t launches() const noexcept
        {
            return m_launches;
        }

    private:
        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{"graphFlow: SourceStage is single-use after a failed batch"};
        }

        void checkStatus(char const* stage)
        {
            StatusCode const code = decodeStatus(m_statusHost.data()[0u]);
            if(code != StatusCode::ok)
            {
                m_failed = true;
                throw BatchError{std::string{stage} + ": device status reports " + statusCodeName(code)};
            }
        }

        TDevice m_device;
        TStorage& m_storage;
        TFunctor m_functor;
        std::uint64_t m_seed = 0u;
        std::uint32_t m_purpose = 0u;
        std::uint32_t m_tileSize;
        std::uint32_t m_frameCap;
        U32View m_status;
        MappedU32View m_statusHost;
        std::uint32_t m_launches = 0u;
        bool m_failed = false;
    };

    /** Transform stage: one batch-wide out-of-place launch, no histogram or scatter. */
    template<typename TDevice, typename TStorage, typename TPayload, typename TFunctor>
    class TransformStage final : public StageBase<TDevice, TStorage>
    {
        using Base = StageBase<TDevice, TStorage>;
        using Access = StorageAccess<TStorage, TPayload>;
        using U32View = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedU32View = ALPAKA_TYPEOF(
            alpaka::onHost::allocMapped<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

    public:
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        TransformStage(
            TDevice const& device,
            TStorage& storage,
            Config const& config,
            TFunctor functor,
            std::uint32_t purpose = 0u)
            : m_device{device}
            , m_storage{storage}
            , m_functor{std::move(functor)}
            , m_seed{config.seed}
            , m_purpose{purpose}
            , m_capacity{storage.chunkCapacity()}
            , m_batchChunks{config.batchChunks}
            , m_tileSize{config.tileSize}
            , m_frameCap{config.frameCap}
            , m_reserveBound{checkedReserveBound(m_capacity, config.batchChunks)}
            , m_inBase{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_inBaseHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_inCount{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_inCountHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_outPosition{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_outPositionHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_batchChunks)})}
            , m_outChunks{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_reserveBound)})}
            , m_outChunksHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_reserveBound)})}
            , m_status{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
            , m_statusHost{alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
        {
            if(m_frameCap == 0u)
                throw std::invalid_argument{"graphFlow: frameCap must be non-zero"};
            m_pending.reserve(m_reserveBound);
        }

        void launch(Queue const& queue, Batch const& batch) override
        {
            throwIfFailed();
            if(batch.chunks.empty())
                throw BatchError{"transform: empty batch"};
            if(batch.chunks.size() > m_batchChunks)
                throw BatchError{
                    "transform: batch of " + std::to_string(batch.chunks.size()) + " chunks exceeds batchChunks "
                    + std::to_string(m_batchChunks)};

            std::uint64_t const particles = batch.particles();
            if(particles == 0u)
                throw BatchError{"transform: empty batch"};
            std::uint32_t const outputChunks = checkedOutputChunks(particles, m_capacity);
            if(outputChunks > m_reserveBound)
                throw BatchError{"transform: exact chunk count exceeds the reservation bound"};

            m_pending.clear();
            std::uint64_t position = 0u;
            for(std::size_t i = 0u; i < batch.chunks.size(); ++i)
            {
                ChunkDescriptor const& chunk = batch.chunks[i];
                m_inBaseHost.data()[i] = static_cast<std::uint32_t>(m_storage.base(chunk.handle));
                m_inCountHost.data()[i] = chunk.fill;
                m_outPositionHost.data()[i] = static_cast<std::uint32_t>(position);
                position += chunk.fill;
            }

            try
            {
                alpaka::onHost::memset(queue, m_status, std::uint8_t{0});
                for(std::uint32_t k = 0u; k < outputChunks; ++k)
                {
                    ChunkHandle const handle = m_storage.acquire();
                    m_outChunksHost.data()[k] = handle.index;
                    std::uint64_t const remaining = particles - static_cast<std::uint64_t>(k) * m_capacity;
                    std::uint32_t const fill
                        = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, m_capacity));
                    m_pending.push_back(ChunkDescriptor{handle, fill, 0u});
                }

                std::uint64_t const chunkCount = batch.chunks.size();
                alpaka::onHost::memcpy(queue, m_inBase, m_inBaseHost, alpaka::Vec{chunkCount});
                alpaka::onHost::memcpy(queue, m_inCount, m_inCountHost, alpaka::Vec{chunkCount});
                alpaka::onHost::memcpy(queue, m_outPosition, m_outPositionHost, alpaka::Vec{chunkCount});
                alpaka::onHost::memcpy(
                    queue,
                    m_outChunks,
                    m_outChunksHost,
                    alpaka::Vec{static_cast<std::uint64_t>(outputChunks)});

                LaunchPolicy const policy = launchPolicy(m_device, particles, m_tileSize, m_frameCap);
                queue.enqueue(
                    policy.frameSpec(),
                    alpaka::KernelBundle{
                        TransformKernel<TFunctor, Access>{m_functor, m_seed, m_purpose},
                        Access::view0(m_storage),
                        Access::view1(m_storage),
                        Access::view2(m_storage),
                        Access::view3(m_storage),
                        Access::view0(m_storage),
                        Access::view1(m_storage),
                        Access::view2(m_storage),
                        Access::view3(m_storage),
                        m_status,
                        m_inBase,
                        m_inCount,
                        m_outPosition,
                        m_outChunks,
                        m_capacity,
                        static_cast<std::uint32_t>(chunkCount)});
                alpaka::onHost::memcpy(queue, m_statusHost, m_status);
            }
            catch(...)
            {
                releasePending();
                m_failed = true;
                throw;
            }

            m_allocated = outputChunks;
            ++m_launches;
        }

        bool advance(Queue const&, Batch const&, std::uint32_t& allocated) override
        {
            throwIfFailed();
            checkStatus("transform");
            allocated = m_allocated;
            m_allocated = 0u;
            return false;
        }

        void finish(Batch const&, std::vector<ChunkDescriptor>& outputs) override
        {
            outputs.insert(outputs.end(), m_pending.begin(), m_pending.end());
            m_pending.clear();
        }

        void abort() override
        {
            releasePending();
            m_allocated = 0u;
            m_failed = true;
        }

        [[nodiscard]] std::uint32_t launches() const noexcept
        {
            return m_launches;
        }

        [[nodiscard]] std::uint32_t reserveBound() const noexcept
        {
            return m_reserveBound;
        }

    private:
        static std::uint32_t checkedReserveBound(std::uint32_t capacity, std::uint32_t batchChunks)
        {
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow: batchChunks must be non-zero"};
            std::uint64_t const particles = static_cast<std::uint64_t>(capacity) * batchChunks;
            std::uint64_t const bound = outputChunkBound(particles, capacity, 1u);
            if(bound > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow: transform batch output-chunk bound overflow"};
            return static_cast<std::uint32_t>(bound);
        }

        static std::uint32_t checkedOutputChunks(std::uint64_t particles, std::uint32_t capacity)
        {
            std::uint64_t const chunks = ceilDivChecked(particles, capacity);
            if(chunks > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow: transform output chunk count overflow"};
            return static_cast<std::uint32_t>(chunks);
        }

        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{"graphFlow: TransformStage is single-use after a failed batch"};
        }

        void checkStatus(char const* stage)
        {
            StatusCode const code = decodeStatus(m_statusHost.data()[0u]);
            if(code != StatusCode::ok)
            {
                m_failed = true;
                throw BatchError{std::string{stage} + ": device status reports " + statusCodeName(code)};
            }
        }

        void releasePending()
        {
            for(ChunkDescriptor const& descriptor : m_pending)
                m_storage.release(descriptor.handle);
            m_pending.clear();
        }

        TDevice m_device;
        TStorage& m_storage;
        TFunctor m_functor;
        std::uint64_t m_seed = 0u;
        std::uint32_t m_purpose = 0u;
        std::uint32_t m_capacity;
        std::uint32_t m_batchChunks;
        std::uint32_t m_tileSize;
        std::uint32_t m_frameCap;
        std::uint32_t m_reserveBound;
        U32View m_inBase;
        MappedU32View m_inBaseHost;
        U32View m_inCount;
        MappedU32View m_inCountHost;
        U32View m_outPosition;
        MappedU32View m_outPositionHost;
        U32View m_outChunks;
        MappedU32View m_outChunksHost;
        U32View m_status;
        MappedU32View m_statusHost;
        std::vector<ChunkDescriptor> m_pending;
        std::uint32_t m_allocated = 0u;
        std::uint32_t m_launches = 0u;
        bool m_failed = false;
    };

    /** Partition stage: one classify and one scatter launch per batch. */
    template<typename TDevice, typename TStorage, typename TPayload, typename TClassifier, typename TUpdate>
    class PartitionStage final : public StageBase<TDevice, TStorage>
    {
        using Base = StageBase<TDevice, TStorage>;
        using Access = StorageAccess<TStorage, TPayload>;
        using Bin = ClassifierBin<TClassifier, TPayload>;
        using U32View = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedU32View = ALPAKA_TYPEOF(
            alpaka::onHost::allocMapped<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using TileView = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<TileDescriptor>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedTileView = ALPAKA_TYPEOF(
            alpaka::onHost::allocMapped<TileDescriptor>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using ClassifyKernel = PartitionClassifyKernel<TClassifier, Access>;
        using ScatterKernel = PartitionScatterKernel<TClassifier, TUpdate, Access>;

    public:
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        PartitionStage(
            TDevice const& device,
            TStorage& storage,
            Config const& config,
            TClassifier classifier,
            TUpdate update,
            std::uint32_t purpose = 0u)
            : m_device{device}
            , m_storage{storage}
            , m_classifier{std::move(classifier)}
            , m_update{std::move(update)}
            , m_seed{config.seed}
            , m_purpose{purpose}
            , m_capacity{storage.chunkCapacity()}
            , m_bins{static_cast<std::uint32_t>(TClassifier::maxBins)}
            , m_batchChunks{config.batchChunks}
            , m_tileSize{config.tileSize}
            , m_frameCap{config.frameCap}
            , m_tileCapacity{checkedTileCapacity(m_capacity, config.tileSize, config.batchChunks)}
            , m_reserveBound{checkedReserveBound(m_capacity, config.batchChunks, m_bins)}
            , m_counts{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_bins) + 1u})}
            , m_countsHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_bins) + 1u})}
            , m_next{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(m_bins)})}
            , m_nextHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_bins)})}
            , m_table{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_bins) + m_reserveBound})}
            , m_tableHost{alpaka::onHost::allocMapped<std::uint32_t>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_bins) + m_reserveBound})}
            , m_tiles{alpaka::onHost::alloc<TileDescriptor>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_tileCapacity)})}
            , m_tilesHost{alpaka::onHost::allocMapped<TileDescriptor>(
                  device,
                  alpaka::Vec{static_cast<std::uint64_t>(m_tileCapacity)})}
            , m_binIds{alpaka::onHost::alloc<std::uint32_t>(
                  device,
                  alpaka::Vec{checkedSlotCount(storage.numChunks(), m_capacity)})}
            , m_status{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
            , m_statusHost{alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
        {
            if(m_frameCap == 0u)
                throw std::invalid_argument{"graphFlow: frameCap must be non-zero"};
            m_totals.assign(m_bins, 0u);
            m_pending.reserve(m_reserveBound);
        }

        void launch(Queue const& queue, Batch const& batch) override
        {
            throwIfFailed();
            if(batch.chunks.empty())
                throw BatchError{"partition: empty batch"};
            if(batch.chunks.size() > m_batchChunks)
                throw BatchError{
                    "partition: batch of " + std::to_string(batch.chunks.size()) + " chunks exceeds batchChunks "
                    + std::to_string(m_batchChunks)};

            m_stage = 0u;
            m_pending.clear();
            m_numTiles = 0u;

            for(ChunkDescriptor const& chunk : batch.chunks)
            {
                std::uint32_t const tiles = tileCount(chunk.fill, m_tileSize);
                for(std::uint32_t tile = 0u; tile < tiles; ++tile)
                {
                    if(m_numTiles >= m_tileCapacity)
                        throw BatchError{"partition: tile descriptor capacity exceeded"};
                    std::uint32_t const tileOffset = tile * m_tileSize;
                    std::uint32_t const remaining = chunk.fill - tileOffset;
                    std::uint32_t const count = remaining < m_tileSize ? remaining : m_tileSize;
                    m_tilesHost.data()[m_numTiles++] = TileDescriptor{chunk.handle.index, tileOffset, count};
                }
            }

            LaunchPolicy const policy = launchPolicy(m_device, batch.particles(), m_tileSize, m_frameCap);
            alpaka::onHost::memset(queue, m_counts, std::uint8_t{0});
            alpaka::onHost::memset(queue, m_status, std::uint8_t{0});
            if(m_numTiles != 0u)
            {
                alpaka::onHost::memcpy(
                    queue,
                    m_tiles,
                    m_tilesHost,
                    alpaka::Vec{static_cast<std::uint64_t>(m_numTiles)});
                auto const tilesView = m_tiles.getSubView(alpaka::Vec{static_cast<std::uint64_t>(m_numTiles)});
                queue.enqueue(
                    policy.frameSpec(),
                    alpaka::KernelBundle{
                        ClassifyKernel{m_classifier, m_seed, m_purpose},
                        Access::view0(m_storage),
                        Access::view1(m_storage),
                        Access::view2(m_storage),
                        Access::view3(m_storage),
                        m_status,
                        m_binIds,
                        m_counts,
                        tilesView,
                        m_capacity});
                ++m_classifyLaunches;
            }
            alpaka::onHost::memcpy(queue, m_statusHost, m_status);
            alpaka::onHost::memcpy(queue, m_countsHost, m_counts);
        }

        bool advance(Queue const& queue, Batch const& batch, std::uint32_t& allocated) override
        {
            throwIfFailed();
            allocated = 0u;
            if(m_stage == 0u)
            {
                planAndScatter(queue, batch, allocated);
                m_stage = 1u;
                return true;
            }
            try
            {
                finalise();
            }
            catch(...)
            {
                releasePending();
                resetBatchState();
                m_failed = true;
                throw;
            }
            m_stage = 0u;
            return false;
        }

        void finish(Batch const&, std::vector<ChunkDescriptor>& outputs) override
        {
            outputs.insert(outputs.end(), m_pending.begin(), m_pending.end());
            m_pending.clear();
        }

        void abort() override
        {
            releasePending();
            resetBatchState();
            m_failed = true;
        }

        [[nodiscard]] std::uint32_t classifyLaunches() const noexcept
        {
            return m_classifyLaunches;
        }

        [[nodiscard]] std::uint32_t scatterLaunches() const noexcept
        {
            return m_scatterLaunches;
        }

        [[nodiscard]] std::uint32_t reserveBound() const noexcept
        {
            return m_reserveBound;
        }

    private:
        static std::uint64_t checkedSlotCount(std::uint32_t numChunks, std::uint32_t capacity)
        {
            std::uint64_t const slots = static_cast<std::uint64_t>(numChunks) * capacity;
            if(slots > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow: partition slot count exceeds the 32-bit slot index"};
            return slots;
        }

        static std::uint32_t checkedTileCapacity(
            std::uint32_t capacity,
            std::uint32_t tileSize,
            std::uint32_t batchChunks)
        {
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow: batchChunks must be non-zero"};
            std::uint64_t const tiles = static_cast<std::uint64_t>(tileCount(capacity, tileSize)) * batchChunks;
            if(tiles > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow: partition batch tile descriptor count overflow"};
            return static_cast<std::uint32_t>(tiles);
        }

        static std::uint32_t checkedReserveBound(std::uint32_t capacity, std::uint32_t batchChunks, std::uint32_t bins)
        {
            if(batchChunks == 0u)
                throw std::invalid_argument{"graphFlow: batchChunks must be non-zero"};
            std::uint64_t const particles = static_cast<std::uint64_t>(capacity) * batchChunks;
            std::uint64_t const bound = outputChunkBound(particles, capacity, bins);
            if(bound > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error{"graphFlow: partition batch output-chunk bound overflow"};
            return static_cast<std::uint32_t>(bound);
        }

        void planAndScatter(Queue const& queue, Batch const& batch, std::uint32_t& allocated)
        {
            std::vector<std::uint32_t> chunkIndices;
            try
            {
                // The classify status copy is enqueued before the batch event, so `advance`
                // observes the status of this batch before it plans any output chunk.
                checkStatus("partition classify");
                LaunchPolicy const policy = launchPolicy(m_device, batch.particles(), m_tileSize, m_frameCap);
                std::uint32_t const* host = m_countsHost.data();
                if(host[m_bins] != 0u)
                {
                    m_failed = true;
                    throw BatchError{"partition classify: device status reports an invalid classifier result"};
                }

                m_totals.assign(m_bins, 0u);
                std::uint64_t particles = 0u;
                for(std::uint32_t bin = 0u; bin < m_bins; ++bin)
                {
                    m_totals[bin] = host[bin];
                    particles += host[bin];
                }
                if(particles != batch.particles())
                {
                    m_failed = true;
                    throw BatchError{"partition classify: classified particle count does not match the batch"};
                }

                DenseBinPlan const plan = planDenseBins(m_totals, m_capacity);
                if(plan.numOutputChunks > m_reserveBound)
                {
                    m_failed = true;
                    throw BatchError{"partition plan: exact chunk count exceeds the reservation bound"};
                }

                chunkIndices.reserve(plan.numOutputChunks);
                m_pending.clear();
                for(std::size_t bin = 0u; bin < m_totals.size(); ++bin)
                {
                    std::uint32_t const chunks = static_cast<std::uint32_t>(ceilDivChecked(m_totals[bin], m_capacity));
                    for(std::uint32_t k = 0u; k < chunks; ++k)
                    {
                        ChunkHandle const handle = m_storage.acquire();
                        chunkIndices.push_back(handle.index);
                        std::uint32_t const fillIndex = plan.binBase[bin] + k;
                        m_pending.push_back(
                            ChunkDescriptor{handle, plan.fills[fillIndex], static_cast<std::uint32_t>(bin)});
                    }
                }

                std::vector<std::uint32_t> const table = buildDenseTable(plan, chunkIndices);
                if(static_cast<std::uint64_t>(table.size()) > m_table.getExtents().x())
                {
                    m_failed = true;
                    throw BatchError{"partition plan: device table buffer too small"};
                }
                std::copy(table.begin(), table.end(), m_tableHost.data());
                alpaka::onHost::memcpy(queue, m_table, m_tableHost);
                alpaka::onHost::memset(queue, m_next, std::uint8_t{0});
                // Reset the status for the scatter stage; its copy follows the scatter kernel.
                alpaka::onHost::memset(queue, m_status, std::uint8_t{0});
                auto const tilesView = m_tiles.getSubView(alpaka::Vec{static_cast<std::uint64_t>(m_numTiles)});

                if(m_numTiles != 0u)
                {
                    queue.enqueue(
                        policy.frameSpec(),
                        alpaka::KernelBundle{
                            ScatterKernel{m_update, m_seed, static_cast<std::uint32_t>(m_purpose + 1u)},
                            Access::view0(m_storage),
                            Access::view1(m_storage),
                            Access::view2(m_storage),
                            Access::view3(m_storage),
                            Access::view0(m_storage),
                            Access::view1(m_storage),
                            Access::view2(m_storage),
                            Access::view3(m_storage),
                            m_status,
                            m_binIds,
                            m_table,
                            m_next,
                            m_counts,
                            tilesView,
                            m_capacity});
                    ++m_scatterLaunches;
                }

                alpaka::onHost::memcpy(queue, m_statusHost, m_status);
                alpaka::onHost::memcpy(queue, m_countsHost, m_counts);
                alpaka::onHost::memcpy(queue, m_nextHost, m_next);
            }
            catch(...)
            {
                releasePending();
                resetBatchState();
                m_failed = true;
                throw;
            }
            allocated = static_cast<std::uint32_t>(chunkIndices.size());
        }

        void resetBatchState()
        {
            m_stage = 0u;
            m_numTiles = 0u;
            std::fill(m_totals.begin(), m_totals.end(), 0u);
            std::fill_n(m_countsHost.data(), static_cast<std::size_t>(m_bins) + 1u, 0u);
        }

        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{"graphFlow: PartitionStage is single-use after a failed batch"};
        }

        void checkStatus(char const* stage)
        {
            StatusCode const code = decodeStatus(m_statusHost.data()[0u]);
            if(code != StatusCode::ok)
            {
                m_failed = true;
                throw BatchError{std::string{stage} + ": device status reports " + statusCodeName(code)};
            }
        }

        void releasePending()
        {
            for(ChunkDescriptor const& descriptor : m_pending)
                m_storage.release(descriptor.handle);
            m_pending.clear();
        }

        void finalise()
        {
            checkStatus("partition scatter");
            if(m_countsHost.data()[m_bins] != 0u)
                throw BatchError{"partition scatter: device status reports a checked updater overflow"};
            for(std::uint32_t bin = 0u; bin < m_bins; ++bin)
            {
                if(m_nextHost.data()[bin] != m_totals[bin])
                    throw BatchError{"partition scatter: next[bin] does not match total[bin]"};
            }
        }

        TDevice m_device;
        TStorage& m_storage;
        TClassifier m_classifier;
        TUpdate m_update;
        std::uint64_t m_seed = 0u;
        std::uint32_t m_purpose = 0u;
        std::uint32_t m_capacity;
        std::uint32_t m_bins;
        std::uint32_t m_batchChunks;
        std::uint32_t m_tileSize;
        std::uint32_t m_frameCap;
        std::uint32_t m_tileCapacity;
        std::uint32_t m_reserveBound;
        U32View m_counts;
        MappedU32View m_countsHost;
        U32View m_next;
        MappedU32View m_nextHost;
        U32View m_table;
        MappedU32View m_tableHost;
        TileView m_tiles;
        MappedTileView m_tilesHost;
        U32View m_binIds;
        U32View m_status;
        MappedU32View m_statusHost;
        std::uint32_t m_stage = 0u;
        std::uint32_t m_numTiles = 0u;
        std::uint32_t m_classifyLaunches = 0u;
        std::uint32_t m_scatterLaunches = 0u;
        std::vector<std::uint32_t> m_totals;
        std::vector<ChunkDescriptor> m_pending;
        bool m_failed = false;
    };

    /** Sink stage: one launch per chunk, no output descriptor. */
    template<typename TDevice, typename TStorage, typename TPayload, typename TFunctor>
    class SinkStage final : public StageBase<TDevice, TStorage>
    {
        using Base = StageBase<TDevice, TStorage>;
        using Access = StorageAccess<TStorage, TPayload>;
        using U32View = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));
        using MappedU32View = ALPAKA_TYPEOF(
            alpaka::onHost::allocMapped<std::uint32_t>(std::declval<TDevice>(), alpaka::Vec{std::uint64_t{1u}}));

    public:
        using Queue = typename Base::Queue;
        using Batch = typename Base::Batch;

        SinkStage(
            TDevice const& device,
            TStorage& storage,
            Config const& config,
            TFunctor functor,
            std::uint32_t purpose = 0u)
            : m_device{device}
            , m_storage{storage}
            , m_functor{std::move(functor)}
            , m_seed{config.seed}
            , m_purpose{purpose}
            , m_tileSize{config.tileSize}
            , m_frameCap{config.frameCap}
            , m_status{alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
            , m_statusHost{alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{1u}})}
        {
            if(m_frameCap == 0u)
                throw std::invalid_argument{"graphFlow: frameCap must be non-zero"};
        }

        void launch(Queue const& queue, Batch const& batch) override
        {
            throwIfFailed();
            ChunkDescriptor const& chunk = batch.chunk();
            alpaka::onHost::memset(queue, m_status, std::uint8_t{0});
            if(chunk.fill != 0u)
            {
                ++m_launches;
                LaunchPolicy const policy = launchPolicy(m_device, chunk.fill, m_tileSize, m_frameCap);
                queue.enqueue(
                    policy.frameSpec(),
                    alpaka::KernelBundle{
                        SinkKernel<TFunctor, Access>{m_functor, m_seed, m_purpose},
                        Access::view0(m_storage),
                        Access::view1(m_storage),
                        Access::view2(m_storage),
                        Access::view3(m_storage),
                        m_status,
                        static_cast<std::uint32_t>(m_storage.base(chunk.handle)),
                        chunk.fill});
            }
            alpaka::onHost::memcpy(queue, m_statusHost, m_status);
        }

        bool advance(Queue const&, Batch const&, std::uint32_t& allocated) override
        {
            throwIfFailed();
            checkStatus("sink");
            allocated = 0u;
            return false;
        }

        void finish(Batch const&, std::vector<ChunkDescriptor>&) override
        {
        }

        void abort() override
        {
            m_failed = true;
        }

        [[nodiscard]] std::uint32_t launches() const noexcept
        {
            return m_launches;
        }

    private:
        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{"graphFlow: SinkStage is single-use after a failed batch"};
        }

        void checkStatus(char const* stage)
        {
            StatusCode const code = decodeStatus(m_statusHost.data()[0u]);
            if(code != StatusCode::ok)
            {
                m_failed = true;
                throw BatchError{std::string{stage} + ": device status reports " + statusCodeName(code)};
            }
        }

        TDevice m_device;
        TStorage& m_storage;
        TFunctor m_functor;
        std::uint64_t m_seed = 0u;
        std::uint32_t m_purpose = 0u;
        std::uint32_t m_tileSize;
        std::uint32_t m_frameCap;
        U32View m_status;
        MappedU32View m_statusHost;
        std::uint32_t m_launches = 0u;
        bool m_failed = false;
    };
} // namespace graphFlow::detail
