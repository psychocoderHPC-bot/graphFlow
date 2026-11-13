/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/* Compile-only check of the typed Flow execution for the HIP backend.
 *
 * There is no AMD device in the validation environment, so this translation unit
 * instantiates the stage and kernel types without running anything. It is not a
 * CTest case and makes no HIP runtime claim.
 */

#include "graphFlow/flow/Execution.hpp"
#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#if !ALPAKA_LANG_HIP
#    error "test_flow_execution_hip.cpp requires a HIP compilation (GRAPHFLOW_ENABLE_HIP=ON)"
#endif

namespace
{
    using Device = alpaka::onHost::Device<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>;
    using Storage = graphFlow::ParticleStorage<Device>;
    using Flow = graphFlow::Flow<graphFlow::Particle, Device>;
    using Access = graphFlow::detail::StorageAccess<Storage, graphFlow::Particle>;

    static_assert(std::same_as<typename Flow::Storage, Storage>);

    struct Generate
    {
        ALPAKA_FN_ACC graphFlow::Particle operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return graphFlow::Particle{id, 0u, 0u, 0u};
        }
    };

    struct Bump
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle& particle) const
        {
            particle.weight += 1u;
        }
    };

    struct Classifier
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            graphFlow::Particle const& particle) const
        {
            return static_cast<std::uint32_t>(particle.id % maxBins);
        }
    };

    struct Update
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            graphFlow::Particle& particle) const
        {
            particle.weight += bin;
            return graphFlow::UpdateResult::success;
        }
    };

    struct Sink
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle const& particle)
            const
        {
            (void) particle;
        }
    };

    using SourceStage = graphFlow::detail::SourceStage<Device, Storage, graphFlow::Particle, Generate>;
    using TransformStage = graphFlow::detail::TransformStage<Device, Storage, graphFlow::Particle, Bump>;
    using PartitionStage = graphFlow::detail::PartitionStage<Device, Storage, graphFlow::Particle, Classifier, Update>;
    using SinkStage = graphFlow::detail::SinkStage<Device, Storage, graphFlow::Particle, Sink>;

    using SourceKernel = graphFlow::detail::SourceKernel<Generate, Access>;
    using TransformKernel = graphFlow::detail::TransformKernel<Bump, Access>;
    using ClassifyKernel = graphFlow::detail::PartitionClassifyKernel<Classifier, Access>;
    using ScatterKernel = graphFlow::detail::PartitionScatterKernel<Classifier, Update, Access>;
    using SinkKernel = graphFlow::detail::SinkKernel<Sink, Access>;

    static_assert(sizeof(SourceStage) > 0u);
    static_assert(sizeof(TransformStage) > 0u);
    static_assert(sizeof(PartitionStage) > 0u);
    static_assert(sizeof(SinkStage) > 0u);
    static_assert(std::is_trivially_copyable_v<SourceKernel>);
    static_assert(std::is_trivially_copyable_v<TransformKernel>);
    static_assert(std::is_trivially_copyable_v<ClassifyKernel>);
    static_assert(std::is_trivially_copyable_v<ScatterKernel>);
    static_assert(std::is_trivially_copyable_v<SinkKernel>);
} // namespace
