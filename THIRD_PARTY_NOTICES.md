<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Third-party notices

graphFlow is licensed under the Mozilla Public License 2.0 (`MPL-2.0`). The
license text is in [`LICENSE`](LICENSE) and
[`LICENSES/MPL-2.0.txt`](LICENSES/MPL-2.0.txt). Copyright 2026 René Widera.

This file lists third-party material that is copied into, vendored in, or
otherwise distributed with this repository. It does not list build tools or
external dependencies that the build consumes from their own installations.

## Vendored source

At commit `e6856b8b0941b422a2069dbc4207fc344f6aa4da` the repository vendors no
third-party source code. `include/graphFlow/thirdparty/` is present as a
documented intake location and is empty apart from its README.

## External build dependencies

These projects are consumed from separate installations or a pinned source
tree. Their source is not redistributed by this repository, so their license
texts are not reproduced here. Consult the revision you build against.

| Component | Version or pin | SPDX identifier |
| --- | --- | --- |
| alpaka | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | `MPL-2.0` |
| Catch2 | 3.4.0 | `BSL-1.0` |

## Concept-level reuse without code copying

graphFlow implements an AoS-over-SoA execution pattern: payload fields stay in
contiguous structure-of-arrays storage, a kernel assembles one
array-of-structs record, mutates it, and scatters it back. The implementation
is original graphFlow code. It was written from that functional description.

The following projects use related particle and execution concepts. None of
their source code is copied, transcribed, or vendored here.

| Project | License family | Use in graphFlow |
| --- | --- | --- |
| spearhed spmacc | LGPL/GPL | Conceptual reference only. No source, header, comment, identifier, or test vector was copied. |
| PIConGPU | LGPL/GPL | Conceptual reference only. No source, header, comment, identifier, or test vector was copied. |

GPL and LGPL source is deliberately excluded from the copied and vendored
material. Combining it with MPL-2.0 files would change the distribution terms
for the combined work, which this project does not do. See
[`docs/licensing.md`](docs/licensing.md) for the full policy.

The earlier internal repository `psychocoderHPC-bot/graph` is reference
material only. Its runtime, scheduler, ABI, RNG, atomics, and warp-pool code
were not ported. This is recorded in [`docs/plan.md`](docs/plan.md).

## llamaLite

llamaLite is not present in this repository. It may only be vendored later
under the following conditions.

- Its license is `MPL-2.0`, which is compatible with this project.
- The vendored copy is unmodified. Reformatting, renaming, and local patches
  are not applied in place.
- The exact upstream commit SHA and repository URL are recorded in
  [`docs/licensing.md`](docs/licensing.md) and in the component README.
- The upstream license file is kept byte-for-byte under
  `include/graphFlow/thirdparty/llamaLite/`, together with the upstream
  notice file.
- A notice is added to this file before the vendoring change is merged.

## Audit rule

No third-party code may enter the tree without a provenance record, an
allowed license, a preserved license file, and a notice in this file. The full
procedure is in [Auditing third-party material](docs/licensing.md#auditing-third-party-material).
The short form is:

1. Record project, upstream URL, exact 40-character commit SHA, SPDX
   identifier, and the SHA-256 of the upstream license file.
2. Copy source only from the allowlist in `docs/licensing.md`. Do not copy GPL,
   LGPL, AGPL, SSPL, or unlicensed code.
3. Vendor the material unmodified under `include/graphFlow/thirdparty/<name>/`
   with its license and a README that states the provenance.
4. Update this file and the inventory in `docs/licensing.md` in the same
   change.
5. Run `reuse lint` and the source-text audit described in
   `docs/licensing.md` before review.
6. Reject the change if any step cannot be evidenced. Remove the material or
   replace it with an independent implementation.

## Reporting

Open an issue in this repository if a notice is missing, a license file is
incomplete, or you believe this repository contains material that originated
elsewhere. Do not include copied upstream source in the report.
