/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace graphFlow
{
    /** Failure classes a device functor may report through `Status`.
     *
     * `ok` is the initial and success value. `invalid` covers rejected application state such as
     * non-finite or impossible geometry, a vacuum ray with no reachable face, a classifier result
     * outside its declared range, or any other checked input failure. `overflow` covers a checked
     * counter or capacity overflow. Both failure codes abort the batch; a capped history is a
     * routed outcome, not a failure.
     */
    enum class StatusCode : std::uint32_t
    {
        ok = 0u,
        invalid = 1u,
        overflow = 2u
    };

    /** Decode a status word. Unknown non-zero values are treated as `invalid`. */
    ALPAKA_FN_ACC inline constexpr StatusCode decodeStatus(std::uint32_t word) noexcept;

    /** Stable diagnostic name of a status code. */
    ALPAKA_FN_ACC inline constexpr char const* statusCodeName(StatusCode code) noexcept;

    /** One device-visible status word exposed to an opting-in public functor.
     *
     * The execution adapters own a single status word per batch. A functor that declares an extra
     * `Status&` argument (after the optional accelerator argument) may call `set()` or `fail()`
     * when it rejects an item. Any non-`ok` value is fatal; the exact failure code is diagnostic.
     * Device code writes the word. The execution adapters copy it to the host after the batch
     * metadata and before observing the batch event, so the host can reject the batch before it
     * plans or publishes outputs.
     *
     * A default-constructed `Status` has no backing word and ignores writes. The execution
     * adapters always pass a bound instance.
     */
    class Status
    {
    public:
        Status() noexcept = default;

        ALPAKA_FN_ACC explicit Status(std::uint32_t* word) noexcept : m_word{word}
        {
        }

        /** Store `code` in the backing word; a null word (concept probing) is a no-op. */
        ALPAKA_FN_ACC void set(StatusCode code) const noexcept
        {
            if(m_word != nullptr)
                *m_word = static_cast<std::uint32_t>(code);
        }

        /** Alias for `set()` with an intent-revealing name. */
        ALPAKA_FN_ACC void fail(StatusCode code) const noexcept
        {
            set(code);
        }

        [[nodiscard]] ALPAKA_FN_ACC StatusCode code() const noexcept
        {
            return m_word == nullptr ? StatusCode::ok : decodeStatus(*m_word);
        }

        [[nodiscard]] ALPAKA_FN_ACC bool ok() const noexcept
        {
            return code() == StatusCode::ok;
        }

    private:
        std::uint32_t* m_word = nullptr;
    };

    /** Decode a status word. Unknown non-zero values are treated as `invalid`. */
    ALPAKA_FN_ACC inline constexpr StatusCode decodeStatus(std::uint32_t word) noexcept
    {
        switch(word)
        {
        case static_cast<std::uint32_t>(StatusCode::ok):
            return StatusCode::ok;
        case static_cast<std::uint32_t>(StatusCode::invalid):
            return StatusCode::invalid;
        case static_cast<std::uint32_t>(StatusCode::overflow):
            return StatusCode::overflow;
        default:
            return StatusCode::invalid;
        }
    }

    /** Stable diagnostic name of a status code. */
    ALPAKA_FN_ACC inline constexpr char const* statusCodeName(StatusCode code) noexcept
    {
        switch(code)
        {
        case StatusCode::ok:
            return "ok";
        case StatusCode::invalid:
            return "invalid";
        case StatusCode::overflow:
            return "overflow";
        }
        return "unknown";
    }
} // namespace graphFlow
