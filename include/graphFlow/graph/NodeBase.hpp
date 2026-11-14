/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/sched/GraphRunner.hpp"

#include <cstdint>
#include <vector>

namespace graphFlow
{
    /** Host-only node interface of the generic graph layer.
     *
     * The virtual calls are made by the host scheduler only. Kernels stay typed and are launched
     * from inside a concrete node; the graph layer never stores a device function pointer. A node is
     * caller-owned and non-owning in `Graph`/`GraphBuilder`: the caller must keep it alive for the
     * lifetime of the graph and of every `GraphRunner` the graph is attached to.
     *
     * The four lifecycle calls map one-to-one onto `GraphRunner::Hooks`:
     *  - `submit`  -> `Hooks::launch`  (enqueue the first device stage),
     *  - `poll`    -> `Hooks::advance` (completion event fired; return true if a further stage was
     *                                   enqueued, report host-acquired output chunks in `allocated`),
     *  - `complete`-> `Hooks::finish`  (publish output descriptors after the last stage),
     *  - `retire`  -> `Hooks::abort`   (release resources of an in-flight batch that will never be
     *                                   published; called only by the runner's teardown path).
     */
    template<typename TDevice>
    class NodeBase
    {
    public:
        using Runner = GraphRunner<TDevice>;
        using Queue = typename Runner::Queue;
        using Batch = typename Runner::Batch;

        virtual ~NodeBase() = default;

        /** Enqueue the first device stage of `batch` on this node's queue. */
        virtual void submit(Queue const& queue, Batch const& batch) = 0;

        /** Called once the node's completion event fired.
         *
         * Return `true` when the call enqueued a further device stage, so the runner re-arms the
         * completion event. A node that has no second stage only needs to return `false`.
         * `allocated` reports output chunks acquired from the pool during host planning; the
         * runner validates it against the batch's reserved output credits.
         */
        virtual bool poll(Queue const&, Batch const&, std::uint32_t& allocated)
        {
            allocated = 0u;
            return false;
        }

        /** Publish the output descriptors of a completed batch. */
        virtual void complete(Batch const& batch, std::vector<ChunkDescriptor>& outputs) = 0;

        /** Release resources acquired by an in-flight batch that will never be published. */
        virtual void retire()
        {
        }
    };

    /** Adapt a host-only `NodeBase` to the existing `GraphRunner::Hooks` lifecycle. */
    template<typename TDevice>
    typename GraphRunner<TDevice>::Hooks makeNodeHooks(NodeBase<TDevice>& node)
    {
        using Runner = GraphRunner<TDevice>;
        using Queue = typename Runner::Queue;
        using Batch = typename Runner::Batch;
        return typename Runner::Hooks{
            [&node](Queue const& queue, Batch const& batch) { node.submit(queue, batch); },
            [&node](Queue const& queue, Batch const& batch, std::uint32_t& allocated)
            { return node.poll(queue, batch, allocated); },
            [&node](Batch const& batch, std::vector<ChunkDescriptor>& outputs) { node.complete(batch, outputs); },
            [&node]() { node.retire(); }};
    }
} // namespace graphFlow
