/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/graphFlow.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

/** Shared AoS-over-SoA test matrix.
 *
 * The helpers are instantiated once for the host device and once for every compiled accelerator
 * backend. All functors carry `ALPAKA_FN_ACC`, and every readback goes through
 * `alpaka::onHost::memcpy`, so the same code runs unchanged on CPU, CUDA, HIP and SYCL.
 */
namespace graphFlow::test::aos
{
    GRAPHFLOW_TAG(Id, idTag);
    GRAPHFLOW_TAG(Value, valueTag);

    using IdField = graphFlow::Field<IdTag, std::uint64_t>;
    using ValueField = graphFlow::Field<ValueTag, std::uint32_t>;
    using Payload = graphFlow::Record<IdField, ValueField>;

    template<typename TDevice>
    using Storage = graphFlow::SoAStorage<TDevice, IdField, ValueField>;

    struct Generate
    {
        ALPAKA_FN_ACC Payload operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            Payload payload{};
            payload[idTag] = id;
            payload[valueTag] = static_cast<std::uint32_t>(id % 3u);
            return payload;
        }
    };

    struct BumpValue
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Payload& payload) const
        {
            payload[valueTag] += 5u;
        }
    };

    using ValueSubset = graphFlow::Set<ValueTag>;
    using ValueSubsetProxy = graphFlow::MutableRecordProxy<Payload, ValueSubset>;

    /** Same bump through the registration-site subset overload; Id stays untouched. */
    struct BumpValueSubset
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, ValueSubsetProxy& proxy) const
        {
            proxy[valueTag] += 5u;
        }
    };

    struct IdMod3
    {
        static constexpr std::uint32_t maxBins = 3u;

        ALPAKA_FN_ACC std::uint32_t operator()([[maybe_unused]] auto const& accelerator, Payload const& payload) const
        {
            return static_cast<std::uint32_t>(payload[idTag] % 3u);
        }
    };

    struct AddBin
    {
        ALPAKA_FN_ACC graphFlow::UpdateResult operator()(
            [[maybe_unused]] auto const& accelerator,
            std::uint32_t bin,
            Payload& payload) const
        {
            payload[valueTag] += bin;
            return graphFlow::UpdateResult::success;
        }
    };

    template<typename TView>
    struct RecordValue
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, Payload const& payload) const
        {
            values[payload[idTag]] = payload[valueTag];
        }
    };

    struct Outcome
    {
        std::vector<std::uint32_t> values;
        graphFlow::SchedulerStats stats;
    };

    inline graphFlow::Tuning tuningFor(std::uint32_t capacity, std::uint32_t tileSize, std::uint32_t batchChunks)
    {
        graphFlow::Tuning tuning;
        tuning.chunkCapacity = capacity;
        tuning.tileSize = tileSize;
        tuning.batchChunks = batchChunks;
        tuning.frameCap = 2u;
        return tuning;
    }

    template<typename TDevice, typename T>
    std::vector<T> readBack(TDevice& device, auto const& view, std::size_t size)
    {
        std::vector<T> host(size);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        alpaka::onHost::memcpy(queue, host, view);
        return host;
    }

    template<typename TDevice, typename TQueue>
    void drain(TDevice& device, TQueue const& queue)
    {
        auto event = device.makeEvent();
        queue.enqueue(event);
        queue.waitFor(event);
        alpaka::onHost::wait(queue);
    }

    template<typename TDevice, typename TStorage>
    void writeRecords(TDevice& device, TStorage& storage, std::uint32_t slot, std::vector<Payload> const& records)
    {
        std::vector<std::uint64_t> ids(records.size());
        std::vector<std::uint32_t> values(records.size());
        for(std::size_t i = 0u; i < records.size(); ++i)
        {
            ids[i] = records[i][idTag];
            values[i] = records[i][valueTag];
        }
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto const extent = alpaka::Vec{static_cast<std::uint64_t>(records.size())};
        auto const offset = alpaka::Vec{static_cast<std::uint64_t>(slot)};
        alpaka::onHost::memcpy(queue, storage.template column<IdTag>().getSubView(offset, extent), ids);
        alpaka::onHost::memcpy(queue, storage.template column<ValueTag>().getSubView(offset, extent), values);
    }

    template<typename TDevice, typename TStorage>
    std::vector<Payload> readRecords(TDevice& device, TStorage& storage, std::uint32_t slot, std::uint32_t count)
    {
        std::vector<std::uint64_t> ids(count);
        std::vector<std::uint32_t> values(count);
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        auto const extent = alpaka::Vec{static_cast<std::uint64_t>(count)};
        auto const offset = alpaka::Vec{static_cast<std::uint64_t>(slot)};
        alpaka::onHost::memcpy(queue, ids, storage.template column<IdTag>().getSubView(offset, extent));
        alpaka::onHost::memcpy(queue, values, storage.template column<ValueTag>().getSubView(offset, extent));
        std::vector<Payload> records(count);
        for(std::uint32_t i = 0u; i < count; ++i)
        {
            records[i][idTag] = ids[i];
            records[i][valueTag] = values[i];
        }
        return records;
    }

    /** End-to-end AoS flow: source, one transform, wide partition, shared sink. */
    template<bool UseSubset, typename TDevice>
    Outcome runFlowImpl(TDevice& device, std::uint64_t particles, std::uint32_t batchChunks)
    {
        constexpr std::uint32_t capacity = 8u;
        std::size_t const size = static_cast<std::size_t>(std::max<std::uint64_t>(particles, 1u));
        auto results = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(size)});
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            std::vector<std::uint32_t> const zero(size, 0u);
            alpaka::onHost::memcpy(queue, results, zero);
        }

        graphFlow::Flow<Payload, TDevice, Storage<TDevice>> flow{device};
        auto source = flow.source("source", Generate{});
        auto bump = [&]()
        {
            if constexpr(UseSubset)
                return flow.template transform<ValueSubset>("bump", BumpValueSubset{});
            else
                return flow.transform("bump", BumpValue{});
        }();
        auto split = flow.partition("split", IdMod3{}, AddBin{});
        using ResultView = decltype(results.getView());
        auto sink = flow.sink("record", RecordValue<ResultView>{results.getView()});
        flow.connect(source, bump);
        flow.connect(bump, split);
        flow.connect(split.template out<0u>(), sink);
        flow.connect(split.template out<1u>(), sink);
        flow.connect(split.template out<2u>(), sink);
        flow.configure(tuningFor(capacity, 4u, batchChunks));

        Outcome outcome;
        outcome.stats = flow.run(source, particles);
        outcome.values = readBack<TDevice, std::uint32_t>(device, results, size);
        outcome.values.resize(static_cast<std::size_t>(particles));
        return outcome;
    }

    template<typename TDevice>
    Outcome runFlow(TDevice& device, std::uint64_t particles, std::uint32_t batchChunks)
    {
        return runFlowImpl<false>(device, particles, batchChunks);
    }

    template<typename TDevice>
    Outcome runSubsetFlow(TDevice& device, std::uint64_t particles, std::uint32_t batchChunks)
    {
        return runFlowImpl<true>(device, particles, batchChunks);
    }

    /** Scalar host oracle of the same source, transform, classifier and update. */
    inline std::vector<std::uint32_t> oracleValues(std::uint64_t particles)
    {
        std::vector<std::uint32_t> values(static_cast<std::size_t>(particles));
        for(std::uint64_t id = 0u; id < particles; ++id)
        {
            std::uint32_t const bin = static_cast<std::uint32_t>(id % 3u);
            values[static_cast<std::size_t>(id)] = bin + 5u + bin;
        }
        return values;
    }

    inline std::map<std::uint32_t, std::uint64_t> histogramOf(std::vector<std::uint32_t> const& values)
    {
        std::map<std::uint32_t, std::uint64_t> histogram;
        for(std::uint32_t value : values)
            ++histogram[value];
        return histogram;
    }

    template<typename TDevice>
    void checkFlowAgainstOracle(TDevice& device, std::uint64_t particles)
    {
        CAPTURE(particles);
        std::vector<std::uint32_t> const expected = oracleValues(particles);
        for(std::uint32_t batchChunks : {1u, 2u})
        {
            CAPTURE(batchChunks);
            Outcome const outcome = runFlow(device, particles, batchChunks);
            REQUIRE(outcome.values.size() == expected.size());
            CHECK(outcome.values == expected);
            CHECK(
                std::accumulate(outcome.values.begin(), outcome.values.end(), std::uint64_t{0u})
                == std::accumulate(expected.begin(), expected.end(), std::uint64_t{0u}));
            CHECK(histogramOf(outcome.values) == histogramOf(expected));
            CHECK(outcome.stats.poolChunks != 0u);
            CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
        }
    }

    template<typename TDevice>
    void checkSubsetFlowAgainstOracle(TDevice& device, std::uint64_t particles)
    {
        CAPTURE(particles);
        std::vector<std::uint32_t> const expected = oracleValues(particles);
        for(std::uint32_t batchChunks : {1u, 2u})
        {
            CAPTURE(batchChunks);
            Outcome const outcome = runSubsetFlow(device, particles, batchChunks);
            REQUIRE(outcome.values.size() == expected.size());
            CHECK(outcome.values == expected);
            CHECK(outcome.stats.poolChunks != 0u);
            CHECK(outcome.stats.peakLiveChunks <= outcome.stats.poolChunks);
        }
    }

    /** The bin-id scratch column is independent of the attribute columns and covers every slot. */
    template<typename TDevice>
    void checkScratchColumn(TDevice& device)
    {
        using StorageType = Storage<TDevice>;
        StorageType storage{device, 2u, 3u};
        std::uint64_t const slots = storage.slotCount();
        REQUIRE(slots == 6u);
        CHECK(StorageType::slotBytes == graphFlow::AttributeList<IdField, ValueField>::slotBytes);
        CHECK(storage.payloadBytes() == slots * StorageType::slotBytes);

        std::vector<Payload> const records{
            Payload{10u, 1u},
            Payload{11u, 2u},
            Payload{12u, 3u},
            Payload{13u, 4u},
            Payload{14u, 5u},
            Payload{15u, 6u}};
        writeRecords(device, storage, 0u, records);

        std::vector<std::uint32_t> binIds(static_cast<std::size_t>(slots));
        for(std::size_t i = 0u; i < binIds.size(); ++i)
            binIds[i] = static_cast<std::uint32_t>(40u + i);
        {
            auto queue = device.makeQueue(alpaka::queueKind::blocking);
            alpaka::onHost::memcpy(queue, storage.binIds(), binIds);
        }

        std::vector<std::uint32_t> const readBinIds
            = readBack<TDevice, std::uint32_t>(device, storage.binIds(), static_cast<std::size_t>(slots));
        CHECK(readBinIds == binIds);

        // Writing the scratch column must not disturb the logical records.
        std::vector<Payload> const roundTrip = readRecords(device, storage, 0u, static_cast<std::uint32_t>(slots));
        REQUIRE(roundTrip.size() == records.size());
        for(std::size_t i = 0u; i < records.size(); ++i)
        {
            CHECK(roundTrip[i][idTag] == records[i][idTag]);
            CHECK(roundTrip[i][valueTag] == records[i][valueTag]);
        }
    }

    /** One launch per stage with AoS storage; a transform that touches one attribute keeps both. */
    template<typename TDevice>
    void checkLaunchCountsAndCopyAll(TDevice& device)
    {
        using StorageType = Storage<TDevice>;
        constexpr std::uint32_t capacity = 4u;
        StorageType storage{device, 16u, capacity};

        graphFlow::Config config;
        config.chunkCapacity = capacity;
        config.batchChunks = 1u;
        config.tileSize = 4u;
        config.frameCap = 1u;
        config.numParticles = capacity;
        config.slotBytes = StorageType::slotBytes;

        using Runner = graphFlow::GraphRunner<TDevice, StorageType>;
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

        {
            graphFlow::ChunkDescriptor const input{storage.acquire(), capacity, 0u};
            graphFlow::detail::SourceStage<TDevice, StorageType, Payload, Generate>
                stage{device, storage, config, Generate{}, 0u};
            typename Runner::Batch const batch{{input}, 0u};
            stage.launch(queue, batch);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            CHECK_FALSE(stage.advance(queue, batch, allocated));
            CHECK(stage.launches() == 1u);

            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(batch, outputs);
            REQUIRE(outputs.size() == 1u);
            std::vector<Payload> const records
                = readRecords(device, storage, static_cast<std::uint32_t>(storage.base(input.handle)), capacity);
            for(std::uint32_t i = 0u; i < capacity; ++i)
            {
                CHECK(records[i][idTag] == i);
                CHECK(records[i][valueTag] == i % 3u);
            }
            storage.release(input.handle);
        }

        {
            auto const input = storage.acquire();
            std::vector<Payload> const seed{
                Payload{1000u, 0u},
                Payload{1001u, 1u},
                Payload{1002u, 2u},
                Payload{1003u, 3u}};
            writeRecords(device, storage, static_cast<std::uint32_t>(storage.base(input)), seed);

            graphFlow::ChunkDescriptor const chunk{input, capacity, 0u};
            graphFlow::detail::TransformStage<TDevice, StorageType, Payload, BumpValue> stage{
                device,
                storage,
                config,
                BumpValue{}};
            typename Runner::Batch const batch{{chunk}, 0u};
            stage.launch(queue, batch);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            CHECK_FALSE(stage.advance(queue, batch, allocated));
            CHECK(stage.launches() == 1u);

            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(batch, outputs);
            REQUIRE(outputs.size() == 1u);
            std::vector<Payload> const records
                = readRecords(device, storage, static_cast<std::uint32_t>(storage.base(outputs[0].handle)), capacity);
            for(std::uint32_t i = 0u; i < capacity; ++i)
            {
                // BumpValue writes only Value; copy-all keeps the untouched Id.
                CHECK(records[i][idTag] == 1000u + i);
                CHECK(records[i][valueTag] == i + 5u);
            }
            for(graphFlow::ChunkDescriptor const& output : outputs)
                storage.release(output.handle);
            storage.release(input);
        }

        {
            auto const input = storage.acquire();
            std::vector<Payload> const seed{
                Payload{2000u, 0u},
                Payload{2001u, 1u},
                Payload{2002u, 2u},
                Payload{2003u, 3u}};
            writeRecords(device, storage, static_cast<std::uint32_t>(storage.base(input)), seed);

            graphFlow::ChunkDescriptor const chunk{input, capacity, 0u};
            graphFlow::detail::TransformStage<TDevice, StorageType, Payload, BumpValueSubset, ValueSubset> stage{
                device,
                storage,
                config,
                BumpValueSubset{}};
            typename Runner::Batch const batch{{chunk}, 0u};
            stage.launch(queue, batch);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            CHECK_FALSE(stage.advance(queue, batch, allocated));
            CHECK(stage.launches() == 1u);

            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(batch, outputs);
            REQUIRE(outputs.size() == 1u);
            std::vector<Payload> const records
                = readRecords(device, storage, static_cast<std::uint32_t>(storage.base(outputs[0].handle)), capacity);
            for(std::uint32_t i = 0u; i < capacity; ++i)
            {
                // The subset functor can only name Value; copy-all keeps Id and the other attributes.
                CHECK(records[i][idTag] == 2000u + i);
                CHECK(records[i][valueTag] == i + 5u);
            }
            for(graphFlow::ChunkDescriptor const& output : outputs)
                storage.release(output.handle);
            storage.release(input);
        }

        {
            auto const input = storage.acquire();
            std::vector<Payload> const seed{Payload{0u, 0u}, Payload{1u, 0u}, Payload{2u, 0u}, Payload{3u, 0u}};
            writeRecords(device, storage, static_cast<std::uint32_t>(storage.base(input)), seed);

            graphFlow::ChunkDescriptor const chunk{input, capacity, 0u};
            graphFlow::detail::PartitionStage<TDevice, StorageType, Payload, IdMod3, AddBin>
                stage{device, storage, config, IdMod3{}, AddBin{}};
            typename Runner::Batch const batch{{chunk}, 0u};
            stage.launch(queue, batch);
            drain(device, queue);
            std::uint32_t allocated = 0u;
            REQUIRE(stage.advance(queue, batch, allocated));
            drain(device, queue);
            CHECK_FALSE(stage.advance(queue, batch, allocated));

            std::vector<graphFlow::ChunkDescriptor> outputs;
            stage.finish(batch, outputs);
            CHECK(stage.classifyLaunches() == 1u);
            CHECK(stage.scatterLaunches() == 1u);

            std::uint32_t total = 0u;
            for(graphFlow::ChunkDescriptor const& output : outputs)
            {
                total += output.fill;
                std::vector<Payload> const records = readRecords(
                    device,
                    storage,
                    static_cast<std::uint32_t>(storage.base(output.handle)),
                    output.fill);
                for(Payload const& record : records)
                {
                    CHECK(record[idTag] % 3u == output.bin);
                    CHECK(record[valueTag] == output.bin);
                }
            }
            CHECK(total == capacity);

            for(graphFlow::ChunkDescriptor const& output : outputs)
                storage.release(output.handle);
            storage.release(input);
        }
    }

    struct DefaultPod
    {
        std::uint64_t id = 0u;
        std::uint32_t value = 0u;
    };

    struct GenerateParticle
    {
        ALPAKA_FN_ACC graphFlow::Particle operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return graphFlow::Particle{id, 0u, 0u, 0u};
        }
    };

    struct BumpParticle
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle& particle) const
        {
            particle.weight += 2u;
            particle.visit += 1u;
        }
    };

    struct GenerateDefaultPod
    {
        ALPAKA_FN_ACC DefaultPod operator()([[maybe_unused]] auto const& accelerator, std::uint64_t id) const
        {
            return DefaultPod{id, 0u};
        }
    };

    struct BumpDefaultPod
    {
        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, DefaultPod& pod) const
        {
            pod.value += 3u;
        }
    };

    template<typename TView>
    struct RecordParticleWeight
    {
        mutable TView weights;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, graphFlow::Particle const& particle)
            const
        {
            weights[particle.id] = particle.weight;
        }
    };

    template<typename TView>
    struct RecordDefaultPod
    {
        mutable TView values;

        ALPAKA_FN_ACC void operator()([[maybe_unused]] auto const& accelerator, DefaultPod const& pod) const
        {
            values[pod.id] = pod.value;
        }
    };

    /** Default storages keep their type, byte size and behavior next to the AoS adapter. */
    template<typename TDevice>
    void checkDefaultStorages(TDevice& device)
    {
        static_assert(std::same_as<
                      typename graphFlow::DefaultStorage<graphFlow::Particle, TDevice>,
                      graphFlow::ParticleStorage<TDevice>>);
        static_assert(std::same_as<
                      typename graphFlow::DefaultStorage<DefaultPod, TDevice>,
                      graphFlow::PayloadSlotStorage<DefaultPod, TDevice>>);
        static_assert(graphFlow::ParticleStorage<TDevice>::slotBytes == graphFlow::payloadBytesPerSlot);
        static_assert(graphFlow::PayloadSlotStorage<DefaultPod, TDevice>::slotBytes == sizeof(DefaultPod));

        constexpr std::uint64_t particles = 17u;
        {
            auto weights
                = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(particles)});
            graphFlow::Flow<graphFlow::Particle, TDevice> flow{device};
            auto source = flow.source("generate", GenerateParticle{});
            auto bump = flow.transform("bump", BumpParticle{});
            using WeightView = decltype(weights.getView());
            auto sink = flow.sink("record", RecordParticleWeight<WeightView>{weights.getView()});
            flow.connect(source, bump);
            flow.connect(bump, sink);
            flow.configure(tuningFor(8u, 4u, 2u));
            auto const stats = flow.run(source, particles);
            CHECK(stats.poolChunks != 0u);

            std::vector<std::uint32_t> const host
                = readBack<TDevice, std::uint32_t>(device, weights, static_cast<std::size_t>(particles));
            for(std::uint32_t weight : host)
                CHECK(weight == 2u);
        }

        {
            auto values
                = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{static_cast<std::uint64_t>(particles)});
            graphFlow::Flow<DefaultPod, TDevice> flow{device};
            auto source = flow.source("generate", GenerateDefaultPod{});
            auto bump = flow.transform("bump", BumpDefaultPod{});
            using ValueView = decltype(values.getView());
            auto sink = flow.sink("record", RecordDefaultPod<ValueView>{values.getView()});
            flow.connect(source, bump);
            flow.connect(bump, sink);
            flow.configure(tuningFor(8u, 4u, 2u));
            auto const stats = flow.run(source, particles);
            CHECK(stats.poolChunks != 0u);

            std::vector<std::uint32_t> const host
                = readBack<TDevice, std::uint32_t>(device, values, static_cast<std::size_t>(particles));
            for(std::uint32_t value : host)
                CHECK(value == 3u);
        }
    }
} // namespace graphFlow::test::aos
