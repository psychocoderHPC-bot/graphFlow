/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "graphFlow/sched/GraphRunner.hpp"

#include <vector>

namespace graphFlow::test
{
    /** Rebuild scheduler hooks with one callable replaced.
     *
     * nvcc 12.6.85 crashes in cudafe++ when non-template test code assigns a lambda to a
     * `std::function` whose signature names `GraphRunner<TDevice, TStorage>::Queue` or `Batch`.
     * Constructing the complete `Hooks` value inside these templates keeps the same callables and
     * the same call order while staying off that frontend path.
     */
    template<typename TDevice, typename TStorage, typename THooks, typename TFn>
    THooks wrapLaunch(THooks hooks, TFn fn)
    {
        using Runner = graphFlow::GraphRunner<TDevice, TStorage>;
        return THooks{
            [base = hooks, fn](
                typename Runner::Queue const& queue,
                typename Runner::Batch const& batch) mutable { fn(base.launch, queue, batch); },
            hooks.advance,
            hooks.finish,
            hooks.abort};
    }

    template<typename TDevice, typename TStorage, typename THooks, typename TFn>
    THooks wrapFinish(THooks hooks, TFn fn)
    {
        using Runner = graphFlow::GraphRunner<TDevice, TStorage>;
        return THooks{
            hooks.launch,
            hooks.advance,
            [base = hooks, fn](
                typename Runner::Batch const& batch,
                std::vector<graphFlow::ChunkDescriptor>& outputs) mutable { fn(base.finish, batch, outputs); },
            hooks.abort};
    }

    template<typename TDevice, typename TStorage, typename THooks, typename TFn>
    THooks wrapAbort(THooks hooks, TFn fn)
    {
        return THooks{
            hooks.launch,
            hooks.advance,
            hooks.finish,
            [base = hooks, fn]() mutable { fn(base.abort); }};
    }
} // namespace graphFlow::test
