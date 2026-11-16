/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <graphFlow/graphFlow.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
    GRAPHFLOW_TAG(Id, idTag);
    GRAPHFLOW_TAG(Value, valueTag);

    using Id = graphFlow::Field<IdTag, std::uint64_t>;
    using Value = graphFlow::Field<ValueTag, std::uint32_t>;
    using Particle = graphFlow::Record<Id, Value>;

    struct Generate
    {
        ALPAKA_FN_ACC Particle operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return Particle{id, static_cast<std::uint32_t>(id)};
        }
    };

    struct Bump
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Particle& particle) const
        {
            particle[valueTag] += 10u;
        }
    };

    struct Mod2
    {
        static constexpr std::uint32_t maxBins = 2u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Particle const& particle)
            const
        {
            return static_cast<std::uint32_t>(particle[idTag] % 2u);
        }
    };

    struct AddBin
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Particle& particle) const
        {
            particle[valueTag] += bin;
            return graphFlow::UpdateResult::success;
        }
    };

    template<typename TView>
    struct RecordValue
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Particle const& particle) const
        {
            values[particle[idTag]] = particle[valueTag];
        }
    };
} // namespace

int main()
{
    auto selector
        = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
    {
        std::cerr << "record_pipeline: no host device available\n";
        return 2;
    }
    auto device = selector.makeDevice(0u);

    constexpr std::uint64_t count = 8u;
    auto values = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{count});
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memset(queue, values, std::uint8_t{0});
    }

    graphFlow::SoAFlow<Particle, decltype(device)> flow{device};
    auto source = flow.source("source", Generate{});
    auto bump = flow.transform("bump", Bump{});
    auto split = flow.partition("split", Mod2{}, AddBin{});
    auto sink = flow.sink("record", RecordValue<decltype(values.getView())>{values.getView()});
    flow.connect(source, bump);
    flow.connect(bump, split);
    flow.connect(split.template out<0u>(), sink);
    flow.connect(split.template out<1u>(), sink);

    graphFlow::Tuning tuning;
    tuning.chunkCapacity = 4u;
    tuning.liveParticles = 8u;
    tuning.tileSize = 4u;
    flow.configure(tuning);
    flow.run(source, count);

    std::vector<std::uint32_t> host(static_cast<std::size_t>(count));
    {
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, values);
    }

    std::uint64_t sum = 0u;
    std::cout << "record_pipeline:";
    for(std::uint32_t value : host)
    {
        std::cout << ' ' << value;
        sum += value;
    }
    std::cout << " sum=" << sum << '\n';
    return sum == 112u ? 0 : 1;
}
