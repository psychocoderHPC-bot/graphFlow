/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "PhotonScattering.hpp"

#if ALPAKA_LANG_ONEAPI
namespace graphFlow::photon
{
    /** Define the double oneAPI photon path in its own SYCL device image.
     *
     * The header only declares the specialization, so the default fp32 callers never instantiate
     * a double kernel. This translation unit is compiled at -O0 for the icpx 2026.2
     * FP64-emulation kernel-argument miscompile described in the header comment.
     */
    template<>
    ScatteringResult runPhotonScatteringDoubleOneApi<OneApiSyclDevice>(
        OneApiSyclDevice device,
        Parameters const& parameters,
        graphFlow::Tuning tuning)
    {
        return runPhotonScatteringAs<double>(std::move(device), parameters, tuning);
    }
} // namespace graphFlow::photon
#endif
