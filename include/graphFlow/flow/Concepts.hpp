/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Config.hpp"
#include "graphFlow/flow/Attributes.hpp"
#include "graphFlow/flow/Payload.hpp"
#include "graphFlow/flow/Status.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace graphFlow
{
    /** Problem marker for the optional accelerator argument of a public functor.
     *
     * The public API does not fix the accumulator type. A functor may accept an accelerator as its
     * first argument (`auto const&` in the documented style) or omit it. The probe deliberately has
     * no members: concept checks only inspect the call expression, never the functor body.
     */
    struct AcceleratorProbe
    {
    };

    /** Outcome of an optional partition updater. */
    enum class UpdateResult
    {
        success,
        overflow,
        /** Checked input failure, for example invalid geometry or an impossible state. */
        invalid
    };

    /** Default updater of a classify-only partition node. */
    struct NoUpdate
    {
        template<typename... TArgs>
        ALPAKA_FN_ACC void operator()(TArgs&&...) const
        {
        }
    };

    namespace detail
    {
        template<typename TFunctor, typename... TArgs>
        concept CallableWithAcceleratorAndEngineAndStatus = requires(
            TFunctor const& functor,
            AcceleratorProbe const& accelerator,
            rng::Engine& engine,
            Status& status,
            TArgs... args) { functor(accelerator, engine, status, args...); };

        template<typename TFunctor, typename... TArgs>
        concept CallableWithAcceleratorAndEngine = requires(
            TFunctor const& functor,
            AcceleratorProbe const& accelerator,
            rng::Engine& engine,
            TArgs... args) { functor(accelerator, engine, args...); };

        template<typename TFunctor, typename... TArgs>
        concept CallableWithAccelerator
            = requires(TFunctor const& functor, AcceleratorProbe const& accelerator, TArgs... args) {
                  functor(accelerator, args...);
              };

        template<typename TFunctor, typename... TArgs>
        concept CallableWithAcceleratorAndStatus
            = requires(TFunctor const& functor, AcceleratorProbe const& accelerator, Status& status, TArgs... args) {
                  functor(accelerator, status, args...);
              };

        template<typename TFunctor, typename... TArgs>
        concept CallableWithStatus
            = requires(TFunctor const& functor, Status& status, TArgs... args) { functor(status, args...); };

        /** Result type of the invocation shape the execution adapters select.
         *
         * The priority matches the dispatcher: accelerator plus status, accelerator, status, bare.
         * The accelerator forms win over the status forms so an existing functor whose first
         * parameter is a generic `auto const&` keeps its accelerator meaning.
         */
        template<typename TFunctor, typename... TArgs>
        constexpr auto functorResultIdentity()
        {
            if constexpr(CallableWithAcceleratorAndEngineAndStatus<TFunctor, TArgs...>)
                return std::type_identity<
                    std::invoke_result_t<TFunctor const&, AcceleratorProbe const&, rng::Engine&, Status&, TArgs...>>{};
            else if constexpr(CallableWithAcceleratorAndEngine<TFunctor, TArgs...>)
                return std::type_identity<
                    std::invoke_result_t<TFunctor const&, AcceleratorProbe const&, rng::Engine&, TArgs...>>{};
            else if constexpr(CallableWithAcceleratorAndStatus<TFunctor, TArgs...>)
                return std::type_identity<
                    std::invoke_result_t<TFunctor const&, AcceleratorProbe const&, Status&, TArgs...>>{};
            else if constexpr(CallableWithAccelerator<TFunctor, TArgs...>)
                return std::type_identity<std::invoke_result_t<TFunctor const&, AcceleratorProbe const&, TArgs...>>{};
            else if constexpr(CallableWithStatus<TFunctor, TArgs...>)
                return std::type_identity<std::invoke_result_t<TFunctor const&, Status&, TArgs...>>{};
            else
                return std::type_identity<std::invoke_result_t<TFunctor const&, TArgs...>>{};
        }

        template<typename TFunctor, typename... TArgs>
        using FunctorResultT = typename decltype(functorResultIdentity<TFunctor, TArgs...>())::type;

        /** A public functor may take the accelerator, a `Status&`, both, or neither. */
        template<typename TFunctor, typename... TArgs>
        concept OptionalContextInvocable
            = CallableWithAcceleratorAndEngineAndStatus<TFunctor, TArgs...>
              || CallableWithAcceleratorAndEngine<TFunctor, TArgs...>
              || CallableWithAcceleratorAndStatus<TFunctor, TArgs...> || CallableWithAccelerator<TFunctor, TArgs...>
              || CallableWithStatus<TFunctor, TArgs...> || std::invocable<TFunctor const&, TArgs...>;
    } // namespace detail

    /** Value usable as a compile-time output-port index: an integral or enumeration bin. */
    template<typename T>
    concept PortValue = std::integral<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>>;

    /** A source functor maps a logical item index to one payload. */
    template<typename TFunctor, typename TPayload>
    concept SourceFor = detail::OptionalContextInvocable<TFunctor, std::uint64_t>
                        && std::convertible_to<detail::FunctorResultT<TFunctor, std::uint64_t>, TPayload>;

    /** Bin type returned by a partition classifier. */
    template<typename TClassifier, typename TPayload>
    using ClassifierBin = std::remove_cvref_t<detail::FunctorResultT<TClassifier, TPayload const&>>;

    /** A partition classifier declares `maxBins` and returns a bin for an immutable payload. */
    template<typename TClassifier, typename TPayload>
    concept PartitionClassifierFor = detail::OptionalContextInvocable<TClassifier, TPayload const&> && requires {
        { TClassifier::maxBins } -> std::convertible_to<std::uint32_t>;
        requires(TClassifier::maxBins > 0u);
        // The dense shared-table limit is a run-time check against the resolved bin path, not a
        // classifier property: the hash path sizes its shared table from the tile width and may
        // declare wider partitions. Keep a public upper bound so the host plan stays bounded.
        requires(TClassifier::maxBins <= Config::maxSupportedBins);
    } && (std::integral<ClassifierBin<TClassifier, TPayload>> || std::is_enum_v<ClassifierBin<TClassifier, TPayload>>) && requires(ClassifierBin<TClassifier, TPayload> bin) {
        { static_cast<std::uint32_t>(bin) } -> std::convertible_to<std::uint32_t>;
    };

    /** An updater receives the selected bin and the mutable payload copy. */
    template<typename TUpdate, typename TBin, typename TPayload>
    concept UpdateFor = detail::OptionalContextInvocable<TUpdate, TBin, TPayload&>
                        && (std::is_void_v<detail::FunctorResultT<TUpdate, TBin, TPayload&>>
                            || std::convertible_to<detail::FunctorResultT<TUpdate, TBin, TPayload&>, UpdateResult>);

    /** A transform functor either mutates the payload or returns its replacement. */
    template<typename TFunctor, typename TPayload>
    concept TransformFor
        = (detail::OptionalContextInvocable<TFunctor, TPayload&>
           && (std::is_void_v<detail::FunctorResultT<TFunctor, TPayload&>>
               || std::same_as<std::remove_cvref_t<detail::FunctorResultT<TFunctor, TPayload&>>, TPayload>) )
          || (detail::OptionalContextInvocable<TFunctor, TPayload const&>
              && (std::is_void_v<detail::FunctorResultT<TFunctor, TPayload const&>>
                  || std::same_as<std::remove_cvref_t<detail::FunctorResultT<TFunctor, TPayload const&>>, TPayload>) );

    /** A payload that is a `graphFlow::Record` and exposes its attribute list. */
    template<typename TPayload>
    concept AttributeRecord = requires {
        typename TPayload::attribute_list;
        typename TPayload::attribute_set;
    };

    /** `TSet` is a valid `Set<Tags...>` for the attributes of `TPayload`. */
    template<typename TPayload, typename TSet>
    concept AttributeSubset = requires {
        typename TPayload::attribute_list;
        requires detail::SetCompatible<TSet, typename TPayload::attribute_list>::value;
    };

    /** A subset transform functor receives a `RecordProxy` limited to `TSet`.
     *
     * The mutable proxy wins over the read-only one, matching the full-record overload. The functor
     * either mutates the proxy or returns a replacement `TPayload`; the stage still loads and stores
     * the complete record, so attributes outside `TSet` keep their values.
     */
    template<typename TFunctor, typename TPayload, typename TSet>
    concept SubsetTransformFor
        = AttributeSubset<TPayload, TSet>
          && ((detail::OptionalContextInvocable<TFunctor, MutableRecordProxy<TPayload, TSet>&>
               && (std::is_void_v<detail::FunctorResultT<TFunctor, MutableRecordProxy<TPayload, TSet>&>>
                   || std::same_as<
                       std::remove_cvref_t<detail::FunctorResultT<TFunctor, MutableRecordProxy<TPayload, TSet>&>>,
                       TPayload>) )
              || (detail::OptionalContextInvocable<TFunctor, ConstRecordProxy<TPayload, TSet> const&>
                  && (std::is_void_v<detail::FunctorResultT<TFunctor, ConstRecordProxy<TPayload, TSet> const&>>
                      || std::same_as<
                          std::remove_cvref_t<
                              detail::FunctorResultT<TFunctor, ConstRecordProxy<TPayload, TSet> const&>>,
                          TPayload>) ));

    /** A sink functor consumes each payload once. */
    template<typename TFunctor, typename TPayload>
    concept SinkFor = detail::OptionalContextInvocable<TFunctor, TPayload const&>;
} // namespace graphFlow
