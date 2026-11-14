/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/graph/Classifier.hpp"
#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>

namespace graphFlow::particle
{
    /** Terminal and feedback outcomes of the A and B partitions. */
    enum class Outcome : std::uint32_t
    {
        sink = 0u,
        feedback = 1u,
        capped = 2u
    };

    /** Source functor: materialise one fresh particle per logical item index.
     *
     * graphFlow injects the engine of the source node; the particle property is its first unsigned
     * output. The functor never handles a seed or a draw offset.
     */
    struct GenerateParticles
    {
        ALPAKA_FN_ACC graphFlow::Particle operator()(
            [[maybe_unused]] auto const& accelerator,
            rng::Engine& engine,
            std::uint64_t id) const
        {
            return graphFlow::Particle{id, engine(), 0u, 0u};
        }
    };

    /** Filter: route to bin 1 when the filter draw reaches the particle property. */
    struct FilterByProperty
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()(
            [[maybe_unused]] auto const& accelerator,
            rng::Engine& engine,
            graphFlow::Particle const& particle) const
        {
            return engine() >= particle.property ? 1u : 0u;
        }
    };

    /** A/B classifier: sink, feedback, or the terminal visit cap. */
    struct SelectOutcome
    {
        static constexpr std::uint32_t maxBins = 3u;

        std::uint32_t maxVisits = 0u;

        ALPAKA_FN_ACC Outcome operator()(
            [[maybe_unused]] auto const& accelerator,
            rng::Engine& engine,
            graphFlow::Particle const& particle) const
        {
            if(engine() < 0x8000'0000u)
                return Outcome::sink;
            return particle.visit < maxVisits ? Outcome::feedback : Outcome::capped;
        }
    };

    /** Feedback updater: add the A/B increment and advance the visit counter once. */
    struct UpdateWeight
    {
        std::uint32_t increment = 0u;

        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            Outcome bin,
            graphFlow::Particle& particle) const
        {
            if(bin != Outcome::feedback)
                return graphFlow::UpdateResult::success;
            if(particle.weight > std::numeric_limits<std::uint32_t>::max() - increment)
                return graphFlow::UpdateResult::overflow;
            particle.weight += increment;
            particle.visit += 1u;
            return graphFlow::UpdateResult::success;
        }
    };

    /** Sink: record the final weight of a particle that leaves the feedback loop. */
    template<typename TWeights>
    struct RecordWeight
    {
        mutable TWeights weights;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle const& particle)
            const
        {
            weights[particle.id] = particle.weight;
        }
    };

    /** Sink: record the final weight and mark a particle stopped by the visit cap. */
    template<typename TWeights, typename TCappedFlags>
    struct RecordCapped
    {
        mutable TWeights weights;
        mutable TCappedFlags cappedFlags;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle const& particle)
            const
        {
            weights[particle.id] = particle.weight;
            cappedFlags[particle.id] = 1u;
        }
    };
} // namespace graphFlow::particle
