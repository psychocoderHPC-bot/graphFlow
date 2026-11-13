/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Umbrella header of the application-facing graphFlow API.
 *
 * The low-level custom-node path (`graphFlow/graph/GraphBuilder.hpp`,
 * `graphFlow/sched/GraphRunner.hpp`, and the hooks adapters) is intentionally not part of this
 * umbrella. Include those headers directly when assembling custom nodes.
 */

#include "graphFlow/flow/Concepts.hpp"
#include "graphFlow/flow/Flow.hpp"
#include "graphFlow/flow/NodeHandle.hpp"
#include "graphFlow/flow/Ports.hpp"
#include "graphFlow/flow/Tuning.hpp"
