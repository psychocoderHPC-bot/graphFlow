/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/rng/ParticleRng.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace graphFlow::test
{
    struct ScalarOutcome
    {
        std::vector<std::uint32_t> weights; // final weight indexed by particle id
        std::uint64_t cappedParticles = 0u;
    };

    /** Scalar oracle for the particle-feedback graph.
     *
     * This is a plain host loop. It shares the RNG mapping with the device policy but nothing else:
     * no chunks, no scheduler, no graph, no device buffers. It exists to detect any routing,
     * ordering or lifetime mistake in the graph implementation.
     */
    inline ScalarOutcome runScalarModel(
        std::uint64_t seed,
        std::uint64_t numParticles,
        std::uint32_t maxVisits)
    {
        ScalarOutcome outcome;
        outcome.weights.resize(static_cast<std::size_t>(numParticles), 0u);

        for(std::uint64_t id = 0u; id < numParticles; ++id)
        {
            std::uint32_t const property = rng::propertyDraw(seed, id);
            std::uint32_t weight = 0u;
            std::uint32_t visit = 0u;

            enum class Stage
            {
                filter,
                processA,
                processB
            };
            Stage stage = Stage::filter;

            while(true)
            {
                if(stage == Stage::filter)
                {
                    std::uint32_t const decision = rng::draw(seed, id, visit, rng::purposeFilter);
                    stage = decision >= property ? Stage::processB : Stage::processA;
                    continue;
                }

                std::uint32_t const purpose = stage == Stage::processA ? rng::purposeProcessA : rng::purposeProcessB;
                std::uint32_t const increment = stage == Stage::processA ? 1000u : 1u;
                bool const survives = rng::draw(seed, id, visit, purpose) >= 0x80000000u;
                if(!survives)
                    break;
                if(visit >= maxVisits)
                {
                    ++outcome.cappedParticles;
                    break;
                }
                if(weight > std::numeric_limits<std::uint32_t>::max() - increment)
                    throw std::overflow_error{"scalar model: weight overflow"};
                weight += increment;
                ++visit;
                stage = Stage::filter;
            }

            outcome.weights[static_cast<std::size_t>(id)] = weight;
        }
        return outcome;
    }
} // namespace graphFlow::test
