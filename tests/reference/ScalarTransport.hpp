/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

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
     * re-implements the four-purpose offset formula, the open-interval uniform and the entire
     * transport loop as a plain host loop with its own Philox wrapper.
     */
    namespace scalarTransport
    {
        enum ScalarEvent : std::uint32_t
        {
            scatter = 0u,
            transmitted = 1u,
            reflected = 2u,
            absorbed = 3u,
            capped = 4u
        };

        inline constexpr std::uint32_t purposeFlight = 0u;
        inline constexpr std::uint32_t purposeCollisionKind = 1u;
        inline constexpr std::uint32_t purposeCosine = 2u;
        inline constexpr std::uint32_t purposeAzimuth = 3u;
        inline constexpr std::uint32_t purposesPerVisit = 4u;

        inline std::uint64_t decisionOffset(std::uint32_t visit, std::uint32_t purpose)
        {
            return 4ull
                * (1ull + static_cast<std::uint64_t>(purposesPerVisit) * visit + purpose);
        }

        inline std::uint32_t drawAt(std::uint64_t seed, std::uint64_t id, std::uint64_t offset)
        {
            alpaka::rand::engine::Philox4x32x10 engine(seed, id, 0u);
            std::uint32_t value = engine();
            for(std::uint64_t i = 0u; i < offset; ++i)
                value = engine();
            return value;
        }

        inline std::uint32_t draw(
            std::uint64_t seed,
            std::uint64_t id,
            std::uint32_t visit,
            std::uint32_t purpose)
        {
            return drawAt(seed, id, decisionOffset(visit, purpose));
        }

        inline double openUniform(std::uint32_t word)
        {
            return (static_cast<double>(word) + 0.5) * (1.0 / 4294967296.0);
        }

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
                    if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(ux)
                       || !std::isfinite(uy) || !std::isfinite(uz) || z < 0.0 || z > thickness)
                        throw std::runtime_error{"scalar transport: non-finite or impossible state"};

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
                    if(faceReachable && (!std::isfinite(faceDistance) || faceDistance < 0.0))
                        throw std::runtime_error{"scalar transport: invalid face distance"};

                    if(total == 0.0)
                    {
                        if(!faceReachable)
                            throw std::runtime_error{"scalar transport: vacuum ray has no reachable face"};
                        x += faceDistance * ux;
                        y += faceDistance * uy;
                        z = faceEvent == transmitted ? thickness : 0.0;
                        terminal = ScalarTerminal{
                            faceEvent,
                            scatterCount,
                            x,
                            y,
                            z,
                            ux,
                            uy,
                            uz};
                        break;
                    }

                    double const flightWord
                        = openUniform(draw(seed, id, scatterCount, purposeFlight));
                    double const sampledDistance = -std::log(flightWord) / total;
                    if(faceReachable && faceDistance <= sampledDistance)
                    {
                        x += faceDistance * ux;
                        y += faceDistance * uy;
                        z = faceEvent == transmitted ? thickness : 0.0;
                        terminal = ScalarTerminal{
                            faceEvent,
                            scatterCount,
                            x,
                            y,
                            z,
                            ux,
                            uy,
                            uz};
                        break;
                    }

                    double const kindWord
                        = openUniform(draw(seed, id, scatterCount, purposeCollisionKind));
                    bool const scatters = kindWord < sigmaS / total;
                    x += sampledDistance * ux;
                    y += sampledDistance * uy;
                    z += sampledDistance * uz;
                    if(!scatters)
                    {
                        terminal = ScalarTerminal{
                            absorbed,
                            scatterCount,
                            x,
                            y,
                            z,
                            ux,
                            uy,
                            uz};
                        break;
                    }
                    if(scatterCount >= maxScatters)
                    {
                        terminal = ScalarTerminal{
                            capped,
                            scatterCount,
                            x,
                            y,
                            z,
                            ux,
                            uy,
                            uz};
                        break;
                    }

                    if(isotropic)
                    {
                        double const mu
                            = 2.0 * openUniform(draw(seed, id, scatterCount, purposeCosine)) - 1.0;
                        double const phi = 2.0 * 3.141592653589793238462643383279502884
                                           * openUniform(draw(seed, id, scatterCount, purposeAzimuth));
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
