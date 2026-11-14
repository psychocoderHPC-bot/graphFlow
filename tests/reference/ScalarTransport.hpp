/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/flow/Debug.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace graphFlow::test
{
    /** Independent scalar photon-transport oracle.
     *
     * The oracle deliberately does not include the example headers or `graphFlow/rng`. It
     * re-implements the packed `(id, event, purpose)` key, constructs its own zero-offset engines,
     * and runs the entire transport loop as a plain host loop. Floating-point draws use the alpaka
     * distributions; there is no local word-to-float conversion.
     */
    namespace scalarTransport
    {
        using Engine = alpaka::rand::engine::Philox4x32x10;

        /** Internal Flow purposes: node 1 is the flight transform, node 3 the scatter transform. */
        inline constexpr std::uint32_t purposeFlight = 2u;
        inline constexpr std::uint32_t purposeScatter = 6u;

        inline constexpr std::uint64_t idMask = 0xFFFF'FFFFull;
        inline constexpr std::uint64_t eventMask = 0xFFFFull;
        inline constexpr std::uint32_t purposeMask = 0xFFFFu;

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

        inline Engine engineFor(std::uint64_t seed, std::uint64_t id, std::uint32_t event, std::uint32_t purpose)
        {
            std::uint64_t const key = mix(packed(id, event, purpose) ^ mix(seed));
            return Engine{key, 0u, 0u};
        }

        enum ScalarEvent : std::uint32_t
        {
            scatter = 0u,
            transmitted = 1u,
            reflected = 2u,
            absorbed = 3u,
            capped = 4u
        };

        struct ScalarTerminal
        {
            std::uint32_t event = absorbed;
            std::uint32_t scatterCount = 0u;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            double uz = 1.0;
        };

        struct ScalarOutcome
        {
            std::vector<ScalarTerminal> terminal; // indexed by photon id
        };

        /** Run one independent scalar history per id. */
        inline ScalarOutcome runScalarTransport(
            std::uint64_t seed,
            std::uint64_t numPhotons,
            double thickness,
            double sigmaS,
            double sigmaA,
            bool isotropic,
            std::uint32_t maxScatters)
        {
            if(!std::isfinite(thickness) || thickness <= 0.0)
                throw std::invalid_argument{"scalar transport: invalid thickness"};
            if(!std::isfinite(sigmaS) || sigmaS < 0.0 || !std::isfinite(sigmaA) || sigmaA < 0.0)
                throw std::invalid_argument{"scalar transport: invalid coefficients"};
            double const total = sigmaS + sigmaA;
            if(!std::isfinite(total))
                throw std::invalid_argument{"scalar transport: non-finite total coefficient"};

            ScalarOutcome outcome;
            outcome.terminal.resize(static_cast<std::size_t>(numPhotons));

            for(std::uint64_t id = 0u; id < numPhotons; ++id)
            {
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                double ux = 0.0;
                double uy = 0.0;
                double uz = 1.0;
                std::uint32_t scatterCount = 0u;
                ScalarTerminal terminal;

                while(true)
                {
                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(ux)
                           || !std::isfinite(uy) || !std::isfinite(uz) || z < 0.0 || z > thickness)
                            throw std::runtime_error{"scalar transport: non-finite or impossible state"};
                    }

                    bool faceReachable = false;
                    double faceDistance = 0.0;
                    std::uint32_t faceEvent = absorbed;
                    if(uz > 0.0)
                    {
                        faceReachable = true;
                        faceDistance = (thickness - z) / uz;
                        faceEvent = transmitted;
                    }
                    else if(uz < 0.0)
                    {
                        faceReachable = true;
                        faceDistance = -z / uz;
                        faceEvent = reflected;
                    }
                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        if(faceReachable && (!std::isfinite(faceDistance) || faceDistance < 0.0))
                            throw std::runtime_error{"scalar transport: invalid face distance"};
                    }

                    if(total == 0.0)
                    {
                        if(!faceReachable)
                            throw std::runtime_error{"scalar transport: vacuum ray has no reachable face"};
                        x += faceDistance * ux;
                        y += faceDistance * uy;
                        z = faceEvent == transmitted ? thickness : 0.0;
                        terminal = ScalarTerminal{faceEvent, scatterCount, x, y, z, ux, uy, uz};
                        break;
                    }

                    Engine flightEngine = engineFor(seed, id, scatterCount, purposeFlight);
                    double const flightUniform = 1.0 - alpaka::rand::distribution::UniformReal<double>{}(flightEngine);
                    double const kindUniform = alpaka::rand::distribution::UniformReal<double>{}(flightEngine);
                    if constexpr(graphFlow::debug::checksEnabled)
                    {
                        if(flightUniform <= 0.0)
                            throw std::runtime_error{"scalar transport: non-positive flight uniform"};
                    }
                    double const sampledDistance = -std::log(flightUniform) / total;
                    if(faceReachable && faceDistance <= sampledDistance)
                    {
                        x += faceDistance * ux;
                        y += faceDistance * uy;
                        z = faceEvent == transmitted ? thickness : 0.0;
                        terminal = ScalarTerminal{faceEvent, scatterCount, x, y, z, ux, uy, uz};
                        break;
                    }

                    bool const scatters = kindUniform < sigmaS / total;
                    x += sampledDistance * ux;
                    y += sampledDistance * uy;
                    z += sampledDistance * uz;
                    if(!scatters)
                    {
                        terminal = ScalarTerminal{absorbed, scatterCount, x, y, z, ux, uy, uz};
                        break;
                    }
                    if(scatterCount >= maxScatters)
                    {
                        terminal = ScalarTerminal{capped, scatterCount, x, y, z, ux, uy, uz};
                        break;
                    }

                    if(isotropic)
                    {
                        Engine scatterEngine = engineFor(seed, id, scatterCount, purposeScatter);
                        constexpr double pi = alpaka::math::constants::pi;
                        double const mu = 2.0 * alpaka::rand::distribution::UniformReal<double>{}(scatterEngine) -1.0;
                        double const phi
                            = 2.0 * pi * alpaka::rand::distribution::UniformReal<double>{}(scatterEngine);
                        double const transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                        ux = transverse * std::cos(phi);
                        uy = transverse * std::sin(phi);
                        uz = mu;
                    }
                    ++scatterCount;
                }

                outcome.terminal[static_cast<std::size_t>(id)] = terminal;
            }
            return outcome;
        }
    } // namespace scalarTransport
} // namespace graphFlow::test
