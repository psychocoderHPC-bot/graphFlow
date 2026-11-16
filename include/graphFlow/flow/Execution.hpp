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
#include "graphFlow/flow/Debug.hpp"
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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

    /** Apply a subset transform functor through a `RecordProxy` over the full local record.
     *
     * The stage loads and stores the complete payload; the proxy only narrows the attribute view the
     * functor can name. Attributes outside the set therefore keep the values loaded from storage.
     */
    template<typename TSet, typename TFunctor, typename TAcc, typename TEngine, typename TStatus, typename TPayload>
    ALPAKA_FN_ACC TPayload applySubsetTransform(
        TFunctor const& functor,
        TAcc const& accelerator,
        TEngine& engine,
        TStatus& status,
        TPayload payload)
    {
        using Proxy = MutableRecordProxy<TPayload, TSet>;
        using ConstProxy = ConstRecordProxy<TPayload, TSet>;
        if constexpr(ContextInvocable<TFunctor, TAcc, TEngine, TStatus, Proxy&>)
        {
            auto proxy = payload.template proxy<TSet>();
            auto invoke
                = [&]() -> decltype(auto) { return invokeWithContext(functor, accelerator, engine, status, proxy); };
            return finishTransform(payload, invoke);
        }
        else
        {
            static_assert(
                ContextInvocable<TFunctor, TAcc, TEngine, TStatus, ConstProxy const&>,
                "graphFlow: subset transform functor must accept the mutable or read-only RecordProxy");
            auto const proxy = std::as_const(payload).template proxy<TSet>();
            auto invoke
                = [&]() -> decltype(auto) { return invokeWithContext(functor, accelerator, engine, status, proxy); };
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

    /** Device-accessible type `KernelBundle` derives from a host view or buffer argument.
     *
     * `onHost::SharedBuffer` becomes its mutable `View`; a view stays a view. The kernels are
     * instantiated over these types, matching the bundle's argument tuple.
     */
    template<typename TView>
    using AccessibleView = std::decay_t<decltype(alpaka::onHost::makeAccessibleOnAcc(std::declval<TView&>()))>;

    /** Load/store trait of a payload storage.
     *
     * The trait names every value view of a storage in declaration order. The array-of-struct
     * default binds four views to the same contiguous payload array; the particle and SoA
     * specializations bind them to ids, properties, weights, visits or to one column per llamaLite
     * attribute. `views()` returns those device-accessible views as a tuple so the launchers can
     * expand them into separate top-level kernel arguments; alpaka only rewrites top-level view
     * arguments for the device.
     *
     * `load` and `store` are templates so they accept both the host view type and the
     * device-accessible view type that `KernelBundle` derives from it. The four-view signatures
     * stay available for callers written against the previous fixed-arity trait, while the
     * `slot`-first forms take one view per attribute in declaration order.
     */
    template<typename TStorage, typename TPayload>
    struct StorageAccess
    {
        using Payload = TPayload;
        using View0 = std::remove_cvref_t<decltype(std::declval<TStorage const&>().payload())>;
        using View1 = View0;
        using View2 = View0;
        using View3 = View0;
        using HostViews = std::tuple<View0, View1, View2, View3>;
        using Views
            = std::tuple<AccessibleView<View0>, AccessibleView<View1>, AccessibleView<View2>, AccessibleView<View3>>;

        static constexpr std::size_t arity = std::tuple_size_v<HostViews>;
        static_assert(arity == 4u, "graphFlow: AoS payload access must expose four views");

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

        [[nodiscard]] static Views views(TStorage& storage)
        {
            return Views{
                alpaka::onHost::makeAccessibleOnAcc(storage.payload()),
                alpaka::onHost::makeAccessibleOnAcc(storage.payload()),
                alpaka::onHost::makeAccessibleOnAcc(storage.payload()),
                alpaka::onHost::makeAccessibleOnAcc(storage.payload())};
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

        template<typename V0, typename... TRest>
        ALPAKA_FN_ACC Payload load(std::uint32_t slot, V0 const& view0, TRest const&...) const
        {
            static_assert(1u + sizeof...(TRest) == arity, "graphFlow: unexpected number of payload views");
            return view0[slot];
        }

        template<typename V0, typename... TRest>
        ALPAKA_FN_ACC void store(std::uint32_t slot, Payload const& payload, V0& view0, TRest&...) const
        {
            static_assert(1u + sizeof...(TRest) == arity, "graphFlow: unexpected number of payload views");
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
        using HostViews = std::tuple<View0, View1, View2, View3>;
        using Views
            = std::tuple<AccessibleView<View0>, AccessibleView<View1>, AccessibleView<View2>, AccessibleView<View3>>;

        static constexpr std::size_t arity = std::tuple_size_v<HostViews>;
        static_assert(arity == 4u, "graphFlow: particle payload access must expose four views");

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

        [[nodiscard]] static Views views(ParticleStorage<TDevice>& storage)
        {
            return Views{
                alpaka::onHost::makeAccessibleOnAcc(storage.ids()),
                alpaka::onHost::makeAccessibleOnAcc(storage.properties()),
                alpaka::onHost::makeAccessibleOnAcc(storage.weights()),
                alpaka::onHost::makeAccessibleOnAcc(storage.visits())};
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

        template<typename VId, typename VProperty, typename VWeight, typename VVisit, typename... TRest>
        ALPAKA_FN_ACC Payload load(
            std::uint32_t slot,
            VId const& ids,
            VProperty const& properties,
            VWeight const& weights,
            VVisit const& visits,
            TRest const&...) const
        {
            static_assert(sizeof...(TRest) == 0u, "graphFlow: particle payload access expects four views");
            return Payload{ids[slot], properties[slot], weights[slot], visits[slot]};
        }

        template<typename VId, typename VProperty, typename VWeight, typename VVisit, typename... TRest>
        ALPAKA_FN_ACC void store(
            std::uint32_t slot,
            Payload const& payload,
            VId& ids,
            VProperty& properties,
            VWeight& weights,
            VVisit& visits,
            TRest&...) const
        {
            static_assert(sizeof...(TRest) == 0u, "graphFlow: particle payload access expects four views");
            ids[slot] = payload.id;
            properties[slot] = payload.property;
            weights[slot] = payload.weight;
            visits[slot] = payload.visit;
        }
    };

    /** SoA access over a llamaLite attribute list.
     *
     * Every attribute column is one view, in declaration order, and the payload is the `Record`
     * over the same fields. The launcher expands `views()` into separate top-level kernel
     * arguments so the device rewrite sees each column on its own.
     */
    template<typename TDevice, typename... TAttrs>
    struct StorageAccess<SoAStorage<TDevice, TAttrs...>, Record<TAttrs...>>
    {
        using Payload = Record<TAttrs...>;
        using Storage = SoAStorage<TDevice, TAttrs...>;
        using AttributeTraits = AttributeList<TAttrs...>;
        using FieldTuple = typename AttributeTraits::field_tuple;

        template<typename TAttr>
        using ColumnView = std::remove_cvref_t<decltype(alpaka::makeView(
            std::declval<Storage&>().template column<typename TAttr::tag_type>()))>;

        using HostViews = std::tuple<ColumnView<TAttrs>...>;
        using Views = std::tuple<AccessibleView<ColumnView<TAttrs>>...>;

        static constexpr std::size_t arity = std::tuple_size_v<HostViews>;
        static_assert(arity == sizeof...(TAttrs), "graphFlow: SoA access must expose one view per attribute");

        [[nodiscard]] static Views views(Storage& storage)
        {
            return Views{alpaka::makeView(storage.template column<typename TAttrs::tag_type>())...};
        }

        template<typename... TViews>
        ALPAKA_FN_ACC Payload load(std::uint32_t slot, TViews const&... views) const
        {
            static_assert(sizeof...(TViews) == arity, "graphFlow: unexpected number of attribute views");
            auto const columns = SoAColumnViewPack<TViews...>{views...};
            return loadImpl(slot, columns, std::index_sequence_for<TAttrs...>{});
        }

        template<typename... TViews>
        ALPAKA_FN_ACC void store(std::uint32_t slot, Payload const& payload, TViews&... views) const
        {
            static_assert(sizeof...(TViews) == arity, "graphFlow: unexpected number of attribute views");
            auto columns = SoAColumnViewPack<TViews...>{views...};
            storeImpl(slot, payload, columns, std::index_sequence_for<TAttrs...>{});
        }

    private:
        template<typename TColumns, std::size_t... Is>
        ALPAKA_FN_ACC static Payload loadImpl(std::uint32_t slot, TColumns const& columns, std::index_sequence<Is...>)
        {
            return Payload{columns.template get<Is>()[slot]...};
        }

        template<typename TColumns, std::size_t... Is>
        ALPAKA_FN_ACC static void storeImpl(
            std::uint32_t slot,
            Payload const& payload,
            TColumns& columns,
            std::index_sequence<Is...>)
        {
            ((columns.template get<Is>()[slot]
              = payload.template get<typename std::tuple_element_t<Is, FieldTuple>::tag_type>()),
             ...);
        }
    };

    namespace enqueue_detail
    {
        template<
            typename TQueue,
            typename TLaunchCfg,
            typename TKernel,
            typename TViews,
            typename TArgs,
            std::size_t... ViewI,
            std::size_t... ArgI>
        void withViews(
            TQueue const& queue,
            TLaunchCfg const& launchCfg,
            TKernel kernel,
            TViews& views,
            TArgs& args,
            std::index_sequence<ViewI...>,
            std::index_sequence<ArgI...>)
        {
            queue.enqueue(
                launchCfg,
                alpaka::KernelBundle{std::move(kernel), std::get<ViewI>(views)..., std::get<ArgI>(args)...});
        }

        template<
            typename TQueue,
            typename TLaunchCfg,
            typename TKernel,
            typename TInViews,
            typename TOutViews,
            typename TArgs,
            std::size_t... InI,
            std::size_t... OutI,
            std::size_t... ArgI>
        void withInOutViews(
            TQueue const& queue,
            TLaunchCfg const& launchCfg,
            TKernel kernel,
            TInViews& inViews,
            TOutViews& outViews,
            TArgs& args,
            std::index_sequence<InI...>,
            std::index_sequence<OutI...>,
            std::index_sequence<ArgI...>)
        {
            queue.enqueue(
                launchCfg,
                alpaka::KernelBundle{
                    std::move(kernel),
                    std::get<InI>(inViews)...,
                    std::get<OutI>(outViews)...,
                    std::get<ArgI>(args)...});
        }
    } // namespace enqueue_detail

    /** Enqueue a kernel with every storage view as a separate top-level argument.
     *
     * Alpaka rewrites only top-level view arguments for the device, so the tuple returned by
     * `StorageAccess::views()` must be expanded at the launch site. The views are device-accessible
     * values, so the bundle keeps them mutable when it copies them.
     */
    template<typename TQueue, typename TLaunchCfg, typename TKernel, typename... TViews, typename... TArgs>
    void enqueueKernelWithViews(
        TQueue const& queue,
        TLaunchCfg const& launchCfg,
        TKernel kernel,
        std::tuple<TViews...> views,
        TArgs&&... args)
    {
        auto argsTuple = std::tuple<std::decay_t<TArgs>...>{std::forward<TArgs>(args)...};
        enqueue_detail::withViews(
            queue,
            launchCfg,
            std::move(kernel),
            views,
            argsTuple,
            std::index_sequence_for<TViews...>{},
            std::index_sequence_for<TArgs...>{});
    }

    /** Enqueue a kernel with distinct input and output view tuples. */
    template<
        typename TQueue,
        typename TLaunchCfg,
        typename TKernel,
        typename... TInViews,
        typename... TOutViews,
        typename... TArgs>
    void enqueueKernelWithInOutViews(
        TQueue const& queue,
        TLaunchCfg const& launchCfg,
        TKernel kernel,
        std::tuple<TInViews...> inViews,
        std::tuple<TOutViews...> outViews,
        TArgs&&... args)
    {
        auto argsTuple = std::tuple<std::decay_t<TArgs>...>{std::forward<TArgs>(args)...};
        enqueue_detail::withInOutViews(
            queue,
            launchCfg,
            std::move(kernel),
            inViews,
            outViews,
            argsTuple,
            std::index_sequence_for<TInViews...>{},
            std::index_sequence_for<TOutViews...>{},
            std::index_sequence_for<TArgs...>{});
    }

    /** Source kernel: one launch materialises one admitted chunk. */
    template<typename TFunctor, typename TAccess, typename TViews = typename TAccess::Views>
    struct SourceKernel;

    template<typename TFunctor, typename TAccess, typename... TViews>
    struct SourceKernel<TFunctor, TAccess, std::tuple<TViews...>>
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        template<alpaka::onAcc::concepts::Acc TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TViews... views,
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
                access.store(base + offset, static_cast<typename TAccess::Payload>(payload), views...);
            }
        }
    };

    /** Out-of-place transform kernel: one launch copies the whole batch.
     *
     * Each block owns one input chunk descriptor and threads stride over its payloads. The output
     * position is the dense concatenation of the batch, so `R = ceil(sum(fill) / C)` output chunks
     * hold the result and no histogram or scatter is involved.
     */
    template<
        typename TFunctor,
        typename TAccess,
        typename TSet = void,
        typename TInViews = typename TAccess::Views,
        typename TOutViews = typename TAccess::Views>
    struct TransformKernel;

    template<typename TFunctor, typename TAccess, typename TSet, typename... TInViews, typename... TOutViews>
    struct TransformKernel<TFunctor, TAccess, TSet, std::tuple<TInViews...>, std::tuple<TOutViews...>>
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        // Explicit parameters keep the abbreviated count below the nvcc 12.6 name-hiding ICE threshold.
        template<
            alpaka::onAcc::concepts::Acc TAcc,
            alpaka::concepts::IMdSpan TStatusView,
            alpaka::concepts::IMdSpan TInBase,
            alpaka::concepts::IMdSpan TInCount,
            alpaka::concepts::IMdSpan TOutPosition,
            alpaka::concepts::IMdSpan TOutChunks>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TInViews... in,
            TOutViews... out,
            TStatusView statusView,
            TInBase inBase,
            TInCount inCount,
            TOutPosition outPosition,
            TOutChunks outChunks,
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
                    typename TAccess::Payload payload = access.load(inSlot + offset, in...);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    if constexpr(std::is_void_v<TSet>)
                        payload = applyTransform(functor, acc, engine, status, std::move(payload));
                    else
                        payload = applySubsetTransform<TSet>(functor, acc, engine, status, std::move(payload));
                    access.store(outSlot, payload, out...);
                }
            }
        }
    };

    /** Out-of-place transform kernel of one input chunk.
     *
     * One launch covers exactly one chunk descriptor: the grid spreads over the chunk's `fill`,
     * every payload is read at `inBase + i` and written at the batch-global output position
     * `position + i`, and a partial tail simply has a smaller `fill`. There is no zero-fill, no
     * neighbour access and no merge with another chunk. The output position math is the batch
     * accounting of the off path: `global / chunkCapacity` selects the output chunk and
     * `global % chunkCapacity` its offset, so the multi-chunk output table stays authoritative.
     */
    template<
        typename TFunctor,
        typename TAccess,
        typename TSet = void,
        typename TInViews = typename TAccess::Views,
        typename TOutViews = typename TAccess::Views>
    struct TransformPerChunkKernel;

    template<typename TFunctor, typename TAccess, typename TSet, typename... TInViews, typename... TOutViews>
    struct TransformPerChunkKernel<TFunctor, TAccess, TSet, std::tuple<TInViews...>, std::tuple<TOutViews...>>
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        template<
            alpaka::onAcc::concepts::Acc TAcc,
            alpaka::concepts::IMdSpan TStatusView,
            alpaka::concepts::IMdSpan TOutChunks>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TInViews... in,
            TOutViews... out,
            TStatusView statusView,
            TOutChunks outChunks,
            std::uint32_t statusSlot,
            std::uint32_t inBase,
            std::uint32_t position,
            std::uint32_t fill,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[statusSlot]};
            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{fill}))
            {
                std::uint64_t const index = static_cast<std::uint64_t>(i);
                if(index >= fill)
                    break;
                std::uint32_t const offset = static_cast<std::uint32_t>(index);
                std::uint32_t const global = position + offset;
                std::uint32_t const outputChunk = global / chunkCapacity;
                std::uint32_t const outputOffset = global % chunkCapacity;
                std::uint32_t const outSlot = outChunks[outputChunk] * chunkCapacity + outputOffset;
                typename TAccess::Payload payload = access.load(inBase + offset, in...);
                rng::Engine engine = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                if constexpr(std::is_void_v<TSet>)
                    payload = applyTransform(functor, acc, engine, status, std::move(payload));
                else
                    payload = applySubsetTransform<TSet>(functor, acc, engine, status, std::move(payload));
                access.store(outSlot, payload, out...);
            }
        }
    };

    /** Dense classify kernel of one typed partition batch. */
    template<
        typename TClassifier,
        typename TAccess,
        std::uint32_t T_B = TClassifier::maxBins,
        typename TViews = typename TAccess::Views>
    struct PartitionClassifyKernel;

    template<typename TClassifier, typename TAccess, std::uint32_t T_B, typename... TViews>
    struct PartitionClassifyKernel<TClassifier, TAccess, T_B, std::tuple<TViews...>>
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");

        TClassifier classifier;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        template<alpaka::onAcc::concepts::Acc TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TViews... views,
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
                    typename TAccess::Payload const payload = access.load(slot, views...);
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
     *
     * `TMode` selects the measured scatter strategy. `SharedStaging` is the historical default;
     * `Direct` skips the shared histogram and reserves one device slot per particle.
     */
    template<
        typename TClassifier,
        typename TUpdate,
        typename TAccess,
        typename TBin = ClassifierBin<TClassifier, typename TAccess::Payload>,
        std::uint32_t T_B = TClassifier::maxBins,
        ScatterMode TMode = ScatterMode::SharedStaging,
        typename TInViews = typename TAccess::Views,
        typename TOutViews = typename TAccess::Views>
    struct PartitionScatterKernel;

    template<
        typename TClassifier,
        typename TUpdate,
        typename TAccess,
        typename TBin,
        std::uint32_t T_B,
        ScatterMode TMode,
        typename... TInViews,
        typename... TOutViews>
    struct PartitionScatterKernel<
        TClassifier,
        TUpdate,
        TAccess,
        TBin,
        T_B,
        TMode,
        std::tuple<TInViews...>,
        std::tuple<TOutViews...>>
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");

        TUpdate update;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        // Explicit parameters keep the abbreviated count below the nvcc 12.6 name-hiding ICE threshold.
        template<
            alpaka::onAcc::concepts::Acc TAcc,
            alpaka::concepts::IMdSpan TStatusView,
            alpaka::concepts::IMdSpan TBinIds,
            alpaka::concepts::IMdSpan TTable,
            alpaka::concepts::IMdSpan TNext,
            alpaka::concepts::IMdSpan TCounts,
            alpaka::concepts::IMdSpan TTiles>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TInViews... in,
            TOutViews... out,
            TStatusView statusView,
            TBinIds binIds,
            TTable table,
            TNext next,
            TCounts counts,
            TTiles tiles,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};

            std::uint32_t const numTiles = static_cast<std::uint32_t>(tiles.getExtents().x());

            if constexpr(TMode == ScatterMode::Direct)
            {
                // One device atomic and one output write per particle. The table lookup is the
                // same exact plan the shared-staging path uses, only the range reservation differs.
                for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
                {
                    std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                    TileDescriptor const tile = tiles[tileIndex];
                    std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                    std::uint32_t const count = tile.count;

                    for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                    {
                        std::uint32_t const slotIn = inBase + static_cast<std::uint32_t>(i);
                        std::uint32_t const bin = binIds[slotIn];
                        std::uint32_t const j
                            = onAcc::atomicAdd(acc, &next[bin], std::uint32_t{1u}, onAcc::scope::device);
                        std::uint32_t const outputChunk = table[std::uint32_t{T_B} + table[bin] + j / chunkCapacity];
                        std::uint32_t const outputOffset = j % chunkCapacity;

                        typename TAccess::Payload payload = access.load(slotIn, in...);
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

                        access.store(outputChunk * chunkCapacity + outputOffset, payload, out...);
                    }
                }
            }
            else
            {
                auto tileCounts
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
                auto tileBase
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});
                auto tileCursor
                    = onAcc::declareSharedMdArray<std::uint32_t, uniqueId()>(acc, CVec<std::uint32_t, T_B>{});

                for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
                {
                    std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                    TileDescriptor const tile = tiles[tileIndex];
                    std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                    std::uint32_t const count = tile.count;

                    for(auto [bin] :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
                        tileCounts[bin] = 0u;
                    onAcc::syncBlockThreads(acc);

                    for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                    {
                        std::uint32_t const bin = binIds[inBase + static_cast<std::uint32_t>(i)];
                        onAcc::atomicAdd(acc, &tileCounts[bin], std::uint32_t{1u}, onAcc::scope::block);
                    }
                    onAcc::syncBlockThreads(acc);

                    for(auto [bin] :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{T_B}}))
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

                        typename TAccess::Payload payload = access.load(slotIn, in...);
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

                        access.store(outputChunk * chunkCapacity + outputOffset, payload, out...);
                    }
                }
            }
        }
    };

    /** Hash classify kernel of one typed partition batch.
     *
     * Mirrors `PartitionClassifyKernel` but replaces the `bins`-word shared histogram with the
     * open-addressing table sized from the tile width. The classifier runs once per payload and
     * the stored bin is reused by the scatter phase. A full table or an out-of-range classifier
     * result writes the status word at `counts[T_B]`, so the host aborts the batch before any
     * output chunk is planned.
     */
    template<
        typename TClassifier,
        typename TAccess,
        typename TBinTable,
        std::uint32_t T_B = TClassifier::maxBins,
        typename TViews = typename TAccess::Views>
    struct PartitionHashClassifyKernel;

    template<typename TClassifier, typename TAccess, typename TBinTable, std::uint32_t T_B, typename... TViews>
    struct PartitionHashClassifyKernel<TClassifier, TAccess, TBinTable, T_B, std::tuple<TViews...>>
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");
        static_assert(TBinTable::isHash, "PartitionHashClassifyKernel requires a hash bin table");

        TClassifier classifier;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        template<alpaka::onAcc::concepts::Acc TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TViews... views,
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
            auto hashTable = onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(
                acc,
                CVec<std::uint32_t, TBinTable::entries>{});
            auto& tableFull = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

            for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
            {
                std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                TileDescriptor const tile = tiles[tileIndex];
                std::uint32_t const base = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                std::uint32_t const count = tile.count;

                for(auto [entry] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{TBinTable::entries}}))
                    hashTable[entry] = HashBinEntry{hashEmptyKey, 0u, 0u, 0u};
                for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{1u}}))
                {
                    alpaka::unused(one);
                    tableFull = 0u;
                }
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const slot = base + static_cast<std::uint32_t>(i);
                    typename TAccess::Payload const payload = access.load(slot, views...);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    std::uint32_t const bin
                        = static_cast<std::uint32_t>(invokeWithContext(classifier, acc, engine, status, payload));
                    if(bin >= T_B)
                    {
                        onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                        continue;
                    }
                    binIds[slot] = bin;
                    if(!hashInsert<ALPAKA_TYPEOF(acc), TBinTable>(acc, hashTable.data(), bin))
                    {
                        onAcc::atomicExch(acc, &tableFull, std::uint32_t{1u}, onAcc::scope::block);
                        break;
                    }
                }
                onAcc::syncBlockThreads(acc);

                if(tableFull != 0u)
                {
                    onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                    break;
                }

                for(auto [entry] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{TBinTable::entries}}))
                {
                    std::uint32_t const bin = hashTable[entry].key;
                    if(bin != hashEmptyKey)
                        onAcc::atomicAdd(acc, &counts[bin], hashTable[entry].count, onAcc::scope::device);
                }
            }
        }
    };

    /** Hash scatter kernel of one typed partition batch.
     *
     * Rebuilds the per-bin counts from the stored `binIds` into the tile-sized hash table,
     * reserves one device-scope range per occupied entry, and applies the updater exactly once per
     * emitted payload. The output table layout is the dense plan `[binBase[B], chunkIndex[R]]`; a
     * full table writes the status word and returns before reserving any output.
     */
    template<
        typename TClassifier,
        typename TUpdate,
        typename TAccess,
        typename TBinTable,
        typename TBin = ClassifierBin<TClassifier, typename TAccess::Payload>,
        std::uint32_t T_B = TClassifier::maxBins,
        typename TInViews = typename TAccess::Views,
        typename TOutViews = typename TAccess::Views>
    struct PartitionHashScatterKernel;

    template<
        typename TClassifier,
        typename TUpdate,
        typename TAccess,
        typename TBinTable,
        typename TBin,
        std::uint32_t T_B,
        typename... TInViews,
        typename... TOutViews>
    struct PartitionHashScatterKernel<
        TClassifier,
        TUpdate,
        TAccess,
        TBinTable,
        TBin,
        T_B,
        std::tuple<TInViews...>,
        std::tuple<TOutViews...>>
    {
        static_assert(T_B > 0u, "a partition needs at least one bin");
        static_assert(TBinTable::isHash, "PartitionHashScatterKernel requires a hash bin table");

        TUpdate update;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        // Explicit parameters keep the abbreviated count below the nvcc 12.6 name-hiding ICE threshold.
        template<
            alpaka::onAcc::concepts::Acc TAcc,
            alpaka::concepts::IMdSpan TStatusView,
            alpaka::concepts::IMdSpan TBinIds,
            alpaka::concepts::IMdSpan TTable,
            alpaka::concepts::IMdSpan TNext,
            alpaka::concepts::IMdSpan TCounts,
            alpaka::concepts::IMdSpan TTiles>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TInViews... in,
            TOutViews... out,
            TStatusView statusView,
            TBinIds binIds,
            TTable table,
            TNext next,
            TCounts counts,
            TTiles tiles,
            std::uint32_t chunkCapacity) const
        {
            using namespace alpaka;
            TAccess access;
            Status status{&statusView[0u]};

            std::uint32_t const numTiles = static_cast<std::uint32_t>(tiles.getExtents().x());
            auto hashTable = onAcc::declareSharedMdArray<HashBinEntry, uniqueId()>(
                acc,
                CVec<std::uint32_t, TBinTable::entries>{});
            auto& tableFull = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

            for(auto const& tileIndexVec : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numTiles}))
            {
                std::uint32_t const tileIndex = static_cast<std::uint32_t>(tileIndexVec.x());
                TileDescriptor const tile = tiles[tileIndex];
                std::uint32_t const inBase = tile.chunkIndex * chunkCapacity + tile.tileOffset;
                std::uint32_t const count = tile.count;

                for(auto [entry] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{TBinTable::entries}}))
                    hashTable[entry] = HashBinEntry{hashEmptyKey, 0u, 0u, 0u};
                for(auto [one] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{1u}}))
                {
                    alpaka::unused(one);
                    tableFull = 0u;
                }
                onAcc::syncBlockThreads(acc);

                // Rebuild the per-bin counts from the stored bin ids; never recompute the classifier.
                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const bin = binIds[inBase + static_cast<std::uint32_t>(i)];
                    if(bin >= T_B)
                    {
                        onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                        continue;
                    }
                    if(!hashInsert<ALPAKA_TYPEOF(acc), TBinTable>(acc, hashTable.data(), bin))
                    {
                        onAcc::atomicExch(acc, &tableFull, std::uint32_t{1u}, onAcc::scope::block);
                        break;
                    }
                }
                onAcc::syncBlockThreads(acc);

                if(tableFull != 0u)
                {
                    onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                    return;
                }

                for(auto [entry] :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{std::uint32_t{TBinTable::entries}}))
                {
                    std::uint32_t const bin = hashTable[entry].key;
                    if(bin == hashEmptyKey)
                        continue;
                    if(bin >= T_B)
                    {
                        onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                        continue;
                    }
                    hashTable[entry].base
                        = onAcc::atomicAdd(acc, &next[bin], hashTable[entry].count, onAcc::scope::device);
                    hashTable[entry].localRank = 0u;
                }
                onAcc::syncBlockThreads(acc);

                for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{count}))
                {
                    std::uint32_t const slotIn = inBase + static_cast<std::uint32_t>(i);
                    std::uint32_t const bin = binIds[slotIn];
                    if(bin >= T_B)
                        continue;
                    HashBinEntry* entry = hashFind<TBinTable>(hashTable.data(), bin);
                    if(entry == nullptr)
                    {
                        onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);
                        continue;
                    }
                    std::uint32_t const rank
                        = onAcc::atomicAdd(acc, &entry->localRank, std::uint32_t{1u}, onAcc::scope::block);
                    std::uint32_t const j = entry->base + rank;
                    std::uint32_t const outputChunk = table[std::uint32_t{T_B} + table[bin] + j / chunkCapacity];
                    std::uint32_t const outputOffset = j % chunkCapacity;

                    typename TAccess::Payload payload = access.load(slotIn, in...);
                    rng::Engine engine
                        = rng::makeEngine(seed, rng::streamId(payload), rng::streamEvent(payload), purpose);
                    UpdateResult const result
                        = applyUpdate(update, acc, engine, status, static_cast<TBin>(bin), payload);
                    if(result == UpdateResult::invalid)
                        status.set(StatusCode::invalid);
                    else if(result == UpdateResult::overflow)
                        status.set(StatusCode::overflow);
                    if(result != UpdateResult::success)
                        onAcc::atomicExch(acc, &counts[std::uint32_t{T_B}], std::uint32_t{1u}, onAcc::scope::device);

                    access.store(outputChunk * chunkCapacity + outputOffset, payload, out...);
                }
            }
        }
    };

    /** Sink kernel: consume one chunk without producing an output descriptor. */
    template<typename TFunctor, typename TAccess, typename TViews = typename TAccess::Views>
    struct SinkKernel;

    template<typename TFunctor, typename TAccess, typename... TViews>
    struct SinkKernel<TFunctor, TAccess, std::tuple<TViews...>>
    {
        TFunctor functor;
        std::uint64_t seed = 0u;
        std::uint32_t purpose = 0u;

        template<alpaka::onAcc::concepts::Acc TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            TViews... views,
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
                typename TAccess::Payload const payload = access.load(base + offset, views...);
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
                enqueueKernelWithViews(
                    queue,
                    policy.frameSpec(),
                    SourceKernel<TFunctor, Access>{m_functor, m_seed, m_purpose},
                    Access::views(m_storage),
                    m_status,
                    batch.firstId,
                    static_cast<std::uint32_t>(m_storage.base(chunk.handle)),
                    chunk.fill);
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

    /** Transform stage: one batch-wide out-of-place launch, no histogram or scatter.
     *
     * `TSet` stays `void` for the full-record overload. A concrete `Set<Tags...>` selects the
     * additive registration-site subset path, which hands the functor a `RecordProxy`.
     */
    template<typename TDevice, typename TStorage, typename TPayload, typename TFunctor, typename TSet = void>
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
            , m_chunkSchedule{config.chunkKernelSchedule}
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
            if(m_chunkSchedule == ChunkKernelSchedule::perChunk)
            {
                // Per-chunk completion slots: one status word per input chunk of a batch, so a
                // fan-out never has two kernels writing the same word. The buffers are allocated
                // only for the opt-in schedule; the off path allocates exactly what it did before.
                m_chunkStatus.emplace(
                    alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{std::uint64_t{m_batchChunks}}));
                m_chunkStatusHost.emplace(
                    alpaka::onHost::allocMapped<std::uint32_t>(device, alpaka::Vec{std::uint64_t{m_batchChunks}}));
            }
        }

        void launch(Queue const& queue, Batch const& batch) override
        {
            throwIfFailed();
            if(m_chunkSchedule == ChunkKernelSchedule::perChunk)
            {
                // The per-chunk path is a separate member; the batch-wide body below is unchanged.
                launchPerChunk(queue, batch);
                return;
            }
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
                auto const views = Access::views(m_storage);
                enqueueKernelWithInOutViews(
                    queue,
                    policy.frameSpec(),
                    TransformKernel<TFunctor, Access, TSet>{m_functor, m_seed, m_purpose},
                    views,
                    views,
                    m_status,
                    m_inBase,
                    m_inCount,
                    m_outPosition,
                    m_outChunks,
                    m_capacity,
                    static_cast<std::uint32_t>(chunkCount));
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

        /** One kernel launch per input chunk, frames sized from that chunk's fill.
         *
         * The host planning, the output-chunk acquisition and the output-position values are the
         * same as in the batch-wide path; only the launch loop differs. Each launch reads its own
         * status slot, so the fan-out has no multi-writer status word.
         */
        void launchPerChunk(Queue const& queue, Batch const& batch)
        {
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
                std::size_t const chunkCount = batch.chunks.size();
                // One status word per input chunk: the memset and the readback bracket the whole
                // fan-out on the same queue, and every launch writes a distinct slot.
                alpaka::onHost::memset(queue, *m_chunkStatus, std::uint8_t{0});
                for(std::uint32_t k = 0u; k < outputChunks; ++k)
                {
                    ChunkHandle const handle = m_storage.acquire();
                    m_outChunksHost.data()[k] = handle.index;
                    std::uint64_t const remaining = particles - static_cast<std::uint64_t>(k) * m_capacity;
                    std::uint32_t const fill
                        = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, m_capacity));
                    m_pending.push_back(ChunkDescriptor{handle, fill, 0u});
                }

                alpaka::onHost::memcpy(
                    queue,
                    m_outChunks,
                    m_outChunksHost,
                    alpaka::Vec{static_cast<std::uint64_t>(outputChunks)});

                auto const views = Access::views(m_storage);
                for(std::size_t i = 0u; i < chunkCount; ++i)
                {
                    ChunkDescriptor const& chunk = batch.chunks[i];
                    if(chunk.fill == 0u)
                        continue;
                    // `m_launches` counts enqueued kernels: one per non-empty chunk here and one
                    // per batch on the off path.
                    ++m_launches;
                    LaunchPolicy const policy = launchPolicy(m_device, chunk.fill, m_tileSize, m_frameCap);
                    enqueueKernelWithInOutViews(
                        queue,
                        policy.frameSpec(),
                        TransformPerChunkKernel<TFunctor, Access, TSet>{m_functor, m_seed, m_purpose},
                        views,
                        views,
                        *m_chunkStatus,
                        m_outChunks,
                        static_cast<std::uint32_t>(i),
                        m_inBaseHost.data()[i],
                        m_outPositionHost.data()[i],
                        chunk.fill,
                        m_capacity);
                }
                alpaka::onHost::memcpy(queue, *m_chunkStatusHost, *m_chunkStatus);
            }
            catch(...)
            {
                releasePending();
                m_failed = true;
                throw;
            }

            m_allocated = outputChunks;
        }

        void throwIfFailed() const
        {
            if(m_failed)
                throw std::logic_error{"graphFlow: TransformStage is single-use after a failed batch"};
        }

        void checkStatus(char const* stage)
        {
            StatusCode const code = m_chunkSchedule == ChunkKernelSchedule::perChunk
                                        ? chunkSlotStatus()
                                        : decodeStatus(m_statusHost.data()[0u]);
            if(code != StatusCode::ok)
            {
                m_failed = true;
                throw BatchError{std::string{stage} + ": device status reports " + statusCodeName(code)};
            }
        }

        /** First failing per-chunk status word, or `ok` when every slot is clean.
         *
         * Slots beyond the committed chunk count stay zero from the memset of every launch, so
         * reading the whole batch-sized buffer is safe and needs no bookkeeping of the last batch.
         */
        StatusCode chunkSlotStatus() const
        {
            for(std::uint32_t i = 0u; i < m_batchChunks; ++i)
            {
                StatusCode const code = decodeStatus(m_chunkStatusHost->data()[i]);
                if(code != StatusCode::ok)
                    return code;
            }
            return StatusCode::ok;
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
        ChunkKernelSchedule m_chunkSchedule = ChunkKernelSchedule::off;
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
        // Per-chunk completion slots; allocated only when the schedule is on.
        std::optional<U32View> m_chunkStatus;
        std::optional<MappedU32View> m_chunkStatusHost;
        std::vector<ChunkDescriptor> m_pending;
        std::uint32_t m_allocated = 0u;
        std::uint32_t m_launches = 0u;
        bool m_failed = false;
    };

    /** Partition stage: one classify and one scatter launch per batch.
     *
     * `TBinTable` selects the dense shared histogram or the tile-sized hash table. `TMode` selects
     * the dense scatter strategy; the hash path always stages and is instantiated with
     * `ScatterMode::SharedStaging`.
     */
    template<
        typename TDevice,
        typename TStorage,
        typename TPayload,
        typename TClassifier,
        typename TUpdate,
        typename TBinTable = DenseBinTable,
        ScatterMode TMode = ScatterMode::SharedStaging>
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
        using DenseClassifyKernel = PartitionClassifyKernel<TClassifier, Access>;
        using DenseScatterKernel
            = PartitionScatterKernel<TClassifier, TUpdate, Access, Bin, TClassifier::maxBins, TMode>;
        using HashClassifyKernel = PartitionHashClassifyKernel<TClassifier, Access, TBinTable>;
        using HashScatterKernel = PartitionHashScatterKernel<TClassifier, TUpdate, Access, TBinTable>;

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
                auto enqueueClassify = [&](auto kernel)
                {
                    enqueueKernelWithViews(
                        queue,
                        policy.frameSpec(),
                        std::move(kernel),
                        Access::views(m_storage),
                        m_status,
                        m_binIds,
                        m_counts,
                        tilesView,
                        m_capacity);
                };
                if constexpr(TBinTable::isHash)
                    enqueueClassify(HashClassifyKernel{m_classifier, m_seed, m_purpose});
                else
                    enqueueClassify(DenseClassifyKernel{m_classifier, m_seed, m_purpose});
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
                    auto const views = Access::views(m_storage);
                    auto enqueueScatter = [&](auto kernel)
                    {
                        enqueueKernelWithInOutViews(
                            queue,
                            policy.frameSpec(),
                            std::move(kernel),
                            views,
                            views,
                            m_status,
                            m_binIds,
                            m_table,
                            m_next,
                            m_counts,
                            tilesView,
                            m_capacity);
                    };
                    if constexpr(TBinTable::isHash)
                        enqueueScatter(
                            HashScatterKernel{m_update, m_seed, static_cast<std::uint32_t>(m_purpose + 1u)});
                    else
                        enqueueScatter(
                            DenseScatterKernel{m_update, m_seed, static_cast<std::uint32_t>(m_purpose + 1u)});
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
                enqueueKernelWithViews(
                    queue,
                    policy.frameSpec(),
                    SinkKernel<TFunctor, Access>{m_functor, m_seed, m_purpose},
                    Access::views(m_storage),
                    m_status,
                    static_cast<std::uint32_t>(m_storage.base(chunk.handle)),
                    chunk.fill);
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
