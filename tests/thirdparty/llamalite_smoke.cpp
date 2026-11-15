/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Host-only smoke test for the vendored llamaLite headers. It exercises the
// public SoA API without alpaka, a device, or the upstream test framework, so a
// broken include export or an unbuildable header fails fast in the CPU suite.

#include <llamaLite/llamaLite.hpp>

#include <cstddef>
#include <cstdio>

namespace
{
    struct momentum_t : llama_lite::TagBase
    {
    };

    struct position_t : llama_lite::TagBase
    {
    };

    inline constexpr momentum_t momentum{};
    inline constexpr position_t position{};

    constexpr std::size_t particles = 8;

    using Record = llama_lite::Record<llama_lite::Field<momentum_t, float>, llama_lite::Field<position_t, double>>;

    using ParticleSoA = llama_lite::SoA<Record, particles>;

    bool check(bool condition, char const* what)
    {
        if(!condition)
        {
            std::fprintf(stderr, "llamaLite smoke failure: %s\n", what);
        }
        return condition;
    }
} // namespace

int main()
{
    static_assert(ParticleSoA::size == particles, "SoA exposes the declared extent");

    ParticleSoA soa;

    for(std::size_t i = 0; i < particles; ++i)
    {
        soa[momentum][i] = static_cast<float>(i) * 0.5F;
        soa[position][i] = static_cast<double>(i) * 0.25;
    }

    bool ok = true;
    for(std::size_t i = 0; i < particles; ++i)
    {
        ok = check(soa[momentum][i] == static_cast<float>(i) * 0.5F, "momentum round trip") && ok;
        ok = check(soa[position][i] == static_cast<double>(i) * 0.25, "position round trip") && ok;
    }

    return ok ? 0 : 1;
}
