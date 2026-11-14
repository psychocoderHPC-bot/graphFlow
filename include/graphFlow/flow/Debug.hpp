/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Compile-time switch for debug-only per-event validation.
 *
 * `GRAPHFLOW_ENABLE_DEBUG_VALIDATION` sets `GRAPHFLOW_DEBUG_VALIDATION` on the `graphFlow` interface
 * target. This header is the only consumer of the macro; physics code uses `checksEnabled` in
 * `if constexpr` and carries no preprocessor branch.
 */
#ifndef GRAPHFLOW_DEBUG_VALIDATION
#    define GRAPHFLOW_DEBUG_VALIDATION 0
#endif

namespace graphFlow::debug
{
    /** True when per-event payload validation is compiled in. */
    inline constexpr bool checksEnabled = GRAPHFLOW_DEBUG_VALIDATION != 0;
} // namespace graphFlow::debug
