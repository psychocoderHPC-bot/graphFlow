/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Compile-time switch for debug-only per-event validation.
 *
 * `GRAPHFLOW_ENABLE_DEBUG_VALIDATION` sets `GRAPHFLOW_DEBUG_VALIDATION` on the `graphFlow` interface
 * target. This header is the only consumer of the macro; physics code and the scatter kernels use
 * `checksEnabled` in `if constexpr` and carry no preprocessor branch.
 *
 * The scatter kernels use the flag for input-conservation counters: every particle of a tile must
 * produce exactly one output write. The counter, its atomics, and the extra block barrier are
 * compiled out in release, so the hot path is identical to the unvalidated kernel.
 */
#ifndef GRAPHFLOW_DEBUG_VALIDATION
#    define GRAPHFLOW_DEBUG_VALIDATION 0
#endif

namespace graphFlow::debug
{
    /** True when per-event payload validation is compiled in. */
    inline constexpr bool checksEnabled = GRAPHFLOW_DEBUG_VALIDATION != 0;
} // namespace graphFlow::debug
