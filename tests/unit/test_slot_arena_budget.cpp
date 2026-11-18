/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "graphFlow/core/Config.hpp"
#include "graphFlow/core/PoolBudget.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

TEST_CASE("the slot-arena budget knob is validated against the reservation", "[slot][arena][config][budget]")
{
    using namespace graphFlow;

    Config config;
    config.chunkCapacity = 4u;
    config.numChunks = 4u;
    config.poolMode = PoolMode::slotArena;
    config.liveParticles = 2u;
    config.poolOutputReserveChunks = 2u;
    std::uint64_t const required = requiredArenaBytes(2u, 4u, 2u);
    CHECK(required == 10u * payloadBytesPerSlot);

    // A larger payload budget with a smaller arena budget is a valid, independent sizing.
    config.poolPayloadBudgetBytes = 4u * required;
    config.slotArenaBudgetBytes = required;
    CHECK_NOTHROW(config.validate());
    CHECK(maxPayloadSlots(config.slotArenaBudgetBytes) == 10u);

    // One byte short is rejected before any allocation, naming the required and available slots.
    config.slotArenaBudgetBytes = required - 1u;
    std::string message;
    try
    {
        config.validate();
    }
    catch(std::invalid_argument const& error)
    {
        message = error.what();
    }
    CHECK(message.find("needs 10 slots") != std::string::npos);
    CHECK(message.find("has 9") != std::string::npos);

    // The arena budget is a sub-budget of the payload budget.
    config.slotArenaBudgetBytes = 5u * required;
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);
}
