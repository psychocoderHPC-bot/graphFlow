<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Third-party source intake

This directory holds vendored third-party source, one subdirectory per
component. It is empty at commit `e6856b8b0941b422a2069dbc4207fc344f6aa4da`.
graphFlow currently has no vendored third-party code.

Read [`docs/licensing.md`](../../../docs/licensing.md) before adding anything
here. [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) is the
user-facing notice file.

## Rules

1. Copy only material whose license is on the allowlist in
   [`docs/licensing.md`](../../../docs/licensing.md#3-check-the-license).
   GPL, LGPL, AGPL, SSPL, and unlicensed source is not copied into this tree.
2. Use one directory per component: `include/graphFlow/thirdparty/<name>/`.
3. Keep the upstream files unmodified. Do not reformat, rename, translate, or
   apply local patches in place. A required fix is recorded as a patch file
   next to the component and documented in its README.
4. Keep the upstream license and notice files inside the component directory,
   byte-for-byte. Record the SHA-256 of the license file.
5. Add a component `README.md` with the upstream URL, the full 40-character
   commit SHA, the SPDX identifier, a file list, and the word "unmodified" or
   an exact list of local changes.
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
    README.md            origin, pinned SHA, license, modification state
    LICENSE              upstream license, unmodified
    ...                  upstream source, unmodified
```

`LICENSES/<SPDX>.txt` is updated when REUSE requires it. The root `LICENSE`
stays the graphFlow MPL-2.0 text.

## llamaLite

llamaLite may be added here later. It is acceptable only as an unmodified
`MPL-2.0` vendored copy with its upstream license, notice file, exact commit
SHA, and a notice in the root file. It is not present today and no placeholder
directory is created for it.
