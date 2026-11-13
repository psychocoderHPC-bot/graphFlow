/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/core/Config.hpp"
#include "graphFlow/flow/Payload.hpp"

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
        overflow
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
        concept CallableWithAccelerator
            = requires(TFunctor const& functor, AcceleratorProbe const& accelerator, TArgs... args) {
                  functor(accelerator, args...);
              };

        template<bool THasAccelerator, typename TFunctor, typename... TArgs>
        struct FunctorResult;

        template<typename TFunctor, typename... TArgs>
        struct FunctorResult<true, TFunctor, TArgs...>
        {
            using type = std::invoke_result_t<TFunctor const&, AcceleratorProbe const&, TArgs...>;
        };

        template<typename TFunctor, typename... TArgs>
        struct FunctorResult<false, TFunctor, TArgs...>
        {
            using type = std::invoke_result_t<TFunctor const&, TArgs...>;
        };

        template<typename TFunctor, typename... TArgs>
        using FunctorResultT =
            typename FunctorResult<CallableWithAccelerator<TFunctor, TArgs...>, TFunctor, TArgs...>::type;

        template<typename TFunctor, typename... TArgs>
        concept OptionalAcceleratorInvocable
            = CallableWithAccelerator<TFunctor, TArgs...> || std::invocable<TFunctor const&, TArgs...>;
    } // namespace detail

    /** Value usable as a compile-time output-port index: an integral or enumeration bin. */
    template<typename T>
    concept PortValue = std::integral<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>>;

    /** A source functor maps a logical item index to one payload. */
    template<typename TFunctor, typename TPayload>
    concept SourceFor = detail::OptionalAcceleratorInvocable<TFunctor, std::uint64_t>
                        && std::convertible_to<detail::FunctorResultT<TFunctor, std::uint64_t>, TPayload>;

    /** Bin type returned by a partition classifier. */
    template<typename TClassifier, typename TPayload>
    using ClassifierBin = std::remove_cvref_t<detail::FunctorResultT<TClassifier, TPayload const&>>;

    /** A partition classifier declares `maxBins` and returns a bin for an immutable payload. */
    template<typename TClassifier, typename TPayload>
    concept PartitionClassifierFor = detail::OptionalAcceleratorInvocable<TClassifier, TPayload const&> && requires {
        { TClassifier::maxBins } -> std::convertible_to<std::uint32_t>;
        requires(TClassifier::maxBins > 0u);
        requires(TClassifier::maxBins <= Config::denseBinLimit);
    } && (std::integral<ClassifierBin<TClassifier, TPayload>> || std::is_enum_v<ClassifierBin<TClassifier, TPayload>>) && requires(ClassifierBin<TClassifier, TPayload> bin) {
        { static_cast<std::uint32_t>(bin) } -> std::convertible_to<std::uint32_t>;
    };

    /** An updater receives the selected bin and the mutable payload copy. */
    template<typename TUpdate, typename TBin, typename TPayload>
    concept UpdateFor = detail::OptionalAcceleratorInvocable<TUpdate, TBin, TPayload&>
                        && (std::is_void_v<detail::FunctorResultT<TUpdate, TBin, TPayload&>>
                            || std::convertible_to<detail::FunctorResultT<TUpdate, TBin, TPayload&>, UpdateResult>);

    /** A transform functor either mutates the payload or returns its replacement. */
    template<typename TFunctor, typename TPayload>
    concept TransformFor
        = (detail::OptionalAcceleratorInvocable<TFunctor, TPayload&>
           && (std::is_void_v<detail::FunctorResultT<TFunctor, TPayload&>>
               || std::same_as<std::remove_cvref_t<detail::FunctorResultT<TFunctor, TPayload&>>, TPayload>) )
          || (detail::OptionalAcceleratorInvocable<TFunctor, TPayload const&>
              && (std::is_void_v<detail::FunctorResultT<TFunctor, TPayload const&>>
                  || std::same_as<std::remove_cvref_t<detail::FunctorResultT<TFunctor, TPayload const&>>, TPayload>) );

    /** A sink functor consumes each payload once. */
    template<typename TFunctor, typename TPayload>
    concept SinkFor = detail::OptionalAcceleratorInvocable<TFunctor, TPayload const&>;
} // namespace graphFlow
