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
    /** Independent engine mapping used by the scalar oracle.
     *
     * The oracle deliberately does not include `graphFlow/rng/ParticleRng.hpp`. It re-implements
     * the documented `(id, event, purpose)` packing, the splitmix64 multiply/xor mixer and the
     * zero-offset engine construction from the plan. Each draw starts a fresh engine, so a defect
     * shared with the production wrapper cannot hide behind a common header.
     */
    namespace scalarRng
    {
        using Engine = alpaka::rand::engine::Philox4x32x10;

        inline constexpr std::uint64_t idMask = 0xFFFF'FFFFull;
        inline constexpr std::uint64_t eventMask = 0xFFFFull;
        inline constexpr std::uint32_t purposeMask = 0xFFFFu;

        // Purpose values assigned by the public Flow example's node order (2 * nodeKey).
        inline constexpr std::uint32_t purposeSource = 0u;
        inline constexpr std::uint32_t purposeFilter = 2u;
        inline constexpr std::uint32_t purposeProcessA = 4u;
        inline constexpr std::uint32_t purposeProcessB = 6u;

        inline constexpr std::uint64_t mix(std::uint64_t value)
        {
            value ^= value >> 30u;
            value *= 0xBF58'476D'1CE4'E5B9ull;
            value ^= value >> 27u;
            value *= 0x94D0'49BB'1331'11EBull;
            value ^= value >> 31u;
            return value;
        }

        inline constexpr std::uint64_t packed(std::uint64_t id, std::uint64_t event, std::uint32_t purpose)
        {
            return (id & idMask) | ((event & eventMask) << 32u)
                   | (static_cast<std::uint64_t>(purpose & purposeMask) << 48u);
        }

        inline Engine engineFor(std::uint64_t seed, std::uint64_t id, std::uint64_t event, std::uint32_t purpose)
        {
            std::uint64_t const key = mix(packed(id, event, purpose) ^ mix(seed));
            return Engine{key, 0u, 0u};
        }

        inline std::uint32_t draw(std::uint64_t seed, std::uint64_t id, std::uint32_t visit, std::uint32_t purpose)
        {
            Engine engine = engineFor(seed, id, visit, purpose);
            return engine();
        }

        inline std::uint32_t propertyDraw(std::uint64_t seed, std::uint64_t id)
        {
            return draw(seed, id, 0u, purposeSource);
        }
    } // namespace scalarRng

    struct ScalarOutcome
    {
        std::vector<std::uint32_t> weights; // final weight indexed by particle id
        std::uint64_t cappedParticles = 0u;
    };

    /** Scalar oracle for the particle-feedback graph.
     *
     * This is a plain host loop. It shares only the documented packed stream mapping with the
     * device policy and has its own Philox wrapper: no chunks, no scheduler, no graph, no device
     * buffers.
     */
    inline ScalarOutcome runScalarModel(std::uint64_t seed, std::uint64_t numParticles, std::uint32_t maxVisits)
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
                bool const survives = scalarRng::draw(seed, id, visit, purpose) >= 0x8000'0000u;
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
