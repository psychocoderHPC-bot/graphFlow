/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace graphFlow::test
{
    /** Independent Philox mapping used by the scalar oracle.
     *
     * The oracle deliberately does not include `graphFlow/rng/ParticleRng.hpp`. The mapping
     * `d = 4 * (1 + 3*visit + purpose)` is re-implemented here from the plan, and the engine is
     * constructed at offset zero and advanced locally. A defect shared between production and
     * oracle therefore cannot hide behind a common header.
     */
    namespace scalarRng
    {
        inline constexpr std::uint32_t purposeProperty = 0u;
        inline constexpr std::uint32_t purposeFilter = 0u;
        inline constexpr std::uint32_t purposeProcessA = 1u;
        inline constexpr std::uint32_t purposeProcessB = 2u;

        inline constexpr std::uint64_t decisionOffset(std::uint32_t visit, std::uint32_t purpose)
        {
            return 4ull * (1ull + 3ull * static_cast<std::uint64_t>(visit) + purpose);
        }

        inline std::uint32_t drawAt(std::uint64_t seed, std::uint64_t particleId, std::uint64_t offset)
        {
            alpaka::rand::engine::Philox4x32x10 engine(seed, particleId, 0u);
            std::uint32_t value = engine();
            for(std::uint64_t i = 0u; i < offset; ++i)
                value = engine();
            return value;
        }

        inline std::uint32_t propertyDraw(std::uint64_t seed, std::uint64_t particleId)
        {
            return drawAt(seed, particleId, 0u);
        }

        inline std::uint32_t draw(
            std::uint64_t seed,
            std::uint64_t particleId,
            std::uint32_t visit,
            std::uint32_t purpose)
        {
            return drawAt(seed, particleId, decisionOffset(visit, purpose));
        }
    } // namespace scalarRng

    struct ScalarOutcome
    {
        std::vector<std::uint32_t> weights; // final weight indexed by particle id
        std::uint64_t cappedParticles = 0u;
    };

    /** Scalar oracle for the particle-feedback graph.
     *
     * This is a plain host loop. It shares only the documented draw-index formula with the device
     * policy and has its own Philox wrapper: no chunks, no scheduler, no graph, no device buffers.
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
            std::uint32_t const property = scalarRng::propertyDraw(seed, id);
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
                    std::uint32_t const decision = scalarRng::draw(seed, id, visit, scalarRng::purposeFilter);
                    stage = decision >= property ? Stage::processB : Stage::processA;
                    continue;
                }

                std::uint32_t const purpose
                    = stage == Stage::processA ? scalarRng::purposeProcessA : scalarRng::purposeProcessB;
                std::uint32_t const increment = stage == Stage::processA ? 1000u : 1u;
                bool const survives = scalarRng::draw(seed, id, visit, purpose) >= 0x80000000u;
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
