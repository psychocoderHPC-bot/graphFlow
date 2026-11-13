/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace graphFlow
{
    /** Logical payload record of the public graph API.
     *
     * A payload is copied by value between stages and may cross the host/device boundary. It must
     * therefore be trivially copyable and default constructible so staging storage can hold it
     * without invoking user code. Field layout is application-owned; graphFlow only moves the
     * record through source, partition, transform, and sink functors.
     */
    template<typename T>
    concept Payload = std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>;
} // namespace graphFlow
