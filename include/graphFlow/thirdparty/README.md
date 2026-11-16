<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Third-party source intake

This directory holds vendored third-party source, one subdirectory per
component. It currently contains `llamaLite/`, an `MPL-2.0` copy from the
spearhed project. That is the only vendored component.

Read [`docs/licensing.md`](../../../docs/licensing.md) before adding anything
here. [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) is the
user-facing notice file.

## Rules

1. Copy only material whose license is on the allowlist in
   [`docs/licensing.md`](../../../docs/licensing.md#3-check-the-license).
   GPL, LGPL, AGPL, SSPL, and unlicensed source is not copied into this tree.
2. Use one directory per component: `include/graphFlow/thirdparty/<name>/`.
3. Keep the upstream files unmodified. Do not reformat, rename, or translate
   them. A required fix goes into a numbered note under the component's
   `PATCHES/` directory, names the affected file, and is written for upstream
   submission. See `llamaLite/PATCHES/` for the pattern.
4. Keep the upstream license and notice files inside the component directory,
   byte-for-byte. Record the SHA-256 of the license file.
5. Add a component provenance file. Use `README.md` when the upstream
   component has none; otherwise add `UPSTREAM.md` and leave the upstream
   `README.md` untouched. It states the upstream URL and path, the full
   40-character commit SHA, the SPDX identifier, the upstream file count, the
   license-file SHA-256, and "unmodified" or an exact list of local changes.
6. Add an entry to [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md)
   and to the inventory in `docs/licensing.md` in the same change.
7. Do not put a graphFlow SPDX header on an upstream file. Its own header
   stays untouched.
8. Run `reuse lint` and the copied-text scan from
   [`docs/licensing.md`](../../../docs/licensing.md#6-scan-for-copied-text).

## Layout

```text
include/graphFlow/thirdparty/
  README.md              this file
  <name>/
    README.md            upstream readme, if the component has one
    UPSTREAM.md          graphFlow provenance record when README.md is upstream
    LICENSE              upstream license, unmodified
    PATCHES/             downstream patch notes, if any
    ...                  upstream source, unmodified except listed patches
```

`LICENSES/<SPDX>.txt` is updated when REUSE requires it. The root `LICENSE`
stays the graphFlow MPL-2.0 text.

## llamaLite

`llamaLite/` is the only vendored component. It comes from spearhed at commit
`aa79f32653452bdaea55d9786c57d1ab10a37b81`, path
`packages/pmacc/libs/llamaLite`, under `MPL-2.0`. The subtree holds the 34
upstream files; `UPSTREAM.md` and `PATCHES/` are graphFlow additions.
`include/llamaLite/Algorithm.hpp` and `include/llamaLite/traits.hpp` carry the
two documented patches, and every other upstream file matches the pinned
commit. The upstream `LICENSE` SHA-256 is
`3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04`.

GraphFlow adapters stay outside the subtree: `include/graphFlow/flow/Attributes.hpp`
and `include/graphFlow/flow/Execution.hpp` consume the headers, and
`tests/thirdparty/llamalite_smoke.cpp` tests the exported interface. See
`llamaLite/UPSTREAM.md` and `llamaLite/PATCHES/README.md` for the full record
and upstream contribution path.
