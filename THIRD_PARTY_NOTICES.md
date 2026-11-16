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

### llamaLite

graphFlow vendors llamaLite from the spearhed project.

| Field | Value |
| --- | --- |
| Upstream | spearhed, <https://github.com/ComputationalRadiationPhysics/spearhed> |
| Upstream path | `packages/pmacc/libs/llamaLite` |
| Pinned commit | `aa79f32653452bdaea55d9786c57d1ab10a37b81` |
| SPDX identifier | `MPL-2.0` |
| Vendored path | `include/graphFlow/thirdparty/llamaLite/` |
| Upstream files | 34, including upstream `LICENSE` and `README.md` |
| Upstream license | `LICENSE`, SHA-256 `3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04` |

The upstream license file is kept byte-for-byte. Every upstream file is
byte-identical to the pinned commit except `include/llamaLite/Algorithm.hpp`
and `include/llamaLite/traits.hpp`. The changes are recorded as
`include/graphFlow/thirdparty/llamaLite/PATCHES/0001-nvcc-12.5-explicit-lambda-template.md`
and
`include/graphFlow/thirdparty/llamaLite/PATCHES/0002-clang-unused-astype-parameter.md`,
with reproductions and removal conditions. `UPSTREAM.md` and `PATCHES/` are
graphFlow additions; the upstream `README.md` stays unmodified.

The patches are written for upstream submission. Report them against
`packages/pmacc/libs/llamaLite` at
<https://github.com/ComputationalRadiationPhysics/spearhed/issues> and send the
fix as a pull request to the same repository. Remove the local patch once
upstream carries an equivalent change.

The vendored subtree is not built as part of graphFlow. graphFlow-owned
adapters and integration code live outside it: `include/graphFlow/flow/Attributes.hpp`
and `include/graphFlow/flow/Execution.hpp` consume the public llamaLite
headers, `tests/thirdparty/llamalite_smoke.cpp` exercises them, and the CMake
target exports the vendored header root. Those files are graphFlow code under
`MPL-2.0`, not upstream llamaLite.

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

llamaLite is an independent header-only library. Only the llamaLite library is
vendored from that repository. No PMacc, spmacc, or PIConGPU source is copied
into the llamaLite subtree. The clean-room boundary covers graphFlow's
classifier, kernels, and adapters. The llamaLite subtree is a separate
`MPL-2.0` component with its own file boundary.

The earlier internal repository `psychocoderHPC-bot/graph` is reference
material only. Its runtime, scheduler, ABI, RNG, atomics, and warp-pool code
were not ported. This is recorded in [`docs/plan.md`](docs/plan.md).

## Audit rule

No third-party code may enter the tree without a provenance record, an
allowed license, a preserved license file, and a notice in this file. The full
procedure is in [Auditing third-party material](docs/licensing.md#auditing-third-party-material).
The short form is:

1. Record project, upstream URL, exact 40-character commit SHA, SPDX
   identifier, and the SHA-256 of the upstream license file.
2. Copy source only from the allowlist in `docs/licensing.md`. Do not copy GPL,
   LGPL, AGPL, SSPL, or unlicensed code.
3. Vendor the upstream files unmodified under
   `include/graphFlow/thirdparty/<name>/` with the license. A downstream patch
   is a separate note under `PATCHES/`, names the affected upstream file, and
   is written for upstream submission.
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
