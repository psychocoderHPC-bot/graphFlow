<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Licensing and provenance

This page states the license of graphFlow, how the project treats
concept-level references, and the rule every third-party addition must pass.
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) is the user-facing notice
file.

## Project license

graphFlow is licensed under the Mozilla Public License 2.0 (`MPL-2.0`).
Copyright 2026 René Widera.

- [`LICENSE`](../LICENSE) holds the license text.
- [`LICENSES/MPL-2.0.txt`](../LICENSES/MPL-2.0.txt) is the REUSE license file.
- [`.reuse/dep5`](../.reuse/dep5) assigns `MPL-2.0` to the tree.
- Source and documentation files carry `SPDX-FileCopyrightText` and
  `SPDX-License-Identifier` headers.
- The `reuse lint` job in `.github/workflows/ci.yml` checks the metadata.

MPL-2.0 is file-level copyleft. A recipient may combine graphFlow files with
other code and distribute the larger work under other terms. Modifications to
graphFlow files must stay under MPL-2.0, and the source of those files must
remain available. The license does not grant trademark rights.

## Clean-room AoS-over-SoA implementation

graphFlow stores particle payload fields in structure-of-arrays (SoA) chunks.
A kernel assembles one array-of-structs (AoS) record from the field arrays,
passes that record to the stage functor, and writes the mutated fields back to
the SoA arrays. `include/graphFlow/graph/Classifier.hpp` and the storage
adapters in `include/graphFlow/flow/` implement this pattern.

This code is a clean-room implementation. It was written from the functional
description of the execution pattern and from the graphFlow design documents,
not by adapting another project's source. The technique itself is a general
data-layout and execution idea. The expression in this repository, including
the type names, interfaces, comments, and tests, was written for graphFlow.

No source file, header, code fragment, comment, identifier, or test vector
from spearhed spmacc or PIConGPU was copied, transcribed, or paraphrased into
graphFlow.

## Copyleft references that are not copied

| Project | License family | Status in this repository |
| --- | --- | --- |
| spearhed spmacc | LGPL/GPL | Not copied, not vendored, not linked. Conceptual reference only. |
| PIConGPU | LGPL/GPL | Not copied, not vendored, not linked. Conceptual reference only. |

The upstream projects are not build dependencies. No file under
`include/`, `examples/`, `tests/`, or `tools/` includes their headers or
reproduces their source text. Their license texts are not reproduced here
because no covered material is distributed.

GPL and LGPL code is excluded from copied material on purpose. Mixing it into
MPL-2.0 source files would impose additional terms on the combined work and
would misstate the license of this repository. A future integration that needs
such code must be proposed to the maintainer as a separate component with its
own license boundary, not merged into an MPL-2.0 file.

The prior internal repository `psychocoderHPC-bot/graph` is also not a source
for this tree. According to [`plan.md`](plan.md), its runtime, scheduler, ABI,
RNG, atomics, and warp-pool code were not ported. Its particle semantics, test
vectors, and recorded limitations informed the design.

## llamaLite

llamaLite is not vendored today. It is accepted only under these conditions.

1. The upstream license is `MPL-2.0`. A different license requires a new
   decision before any copy is made.
2. The vendored tree is unmodified. Do not reformat, rename, or apply local
   edits inside the copied files.
3. The upstream repository URL and the exact 40-character commit SHA are
   recorded in the inventory below and in the component README.
4. The upstream license file is copied byte-for-byte, and the SHA-256 of the
   upstream license file is recorded. Upstream notice files are copied too.
5. The component README states the origin, the pinned SHA, the license, and
   whether anything in the directory differs from upstream. It must say
   "unmodified" when it does.
6. `THIRD_PARTY_NOTICES.md` gains the notice in the same change.
7. `reuse lint` and the source-text audit pass.

## Inventory

At base commit `e6856b8b0941b422a2069dbc4207fc344f6aa4da`:

| Component | Origin | Pinned SHA | License | Vendored | Modified |
| --- | --- | --- | --- | --- | --- |
| graphFlow | this repository | n/a | `MPL-2.0` | n/a | n/a |
| alpaka | external dependency | `eb677e8d7f70d79b35df25d9002f94ba669b6f44` | `MPL-2.0` | no | no |
| Catch2 | external dependency | 3.4.0 release | `BSL-1.0` | no | no |
| spearhed spmacc | concept reference | none | LGPL/GPL | no | no |
| PIConGPU | concept reference | none | LGPL/GPL | no | no |
| llamaLite | not present | none | `MPL-2.0` if later vendored | no | no |

A vendored component is redistributed source. An external dependency is built
from its own installation and is not part of this repository's source
distribution.

## Auditing third-party material

Run this audit before a change that adds, updates, or removes vendored code,
and before a release. Record the result in the pull request.

### 1. Identify the material

List every copied file and its origin. A file is third-party if it was written
outside this repository, even when it is small. Generated files count when
their generator or input came from outside.

### 2. Record provenance

For every component, record:

- project name and upstream URL;
- exact 40-character commit SHA, not a branch or tag;
- SPDX identifier from the upstream license file;
- path and SHA-256 of the upstream license file;
- upstream notice file, if any;
- list of copied files and whether each is modified.

### 3. Check the license

Copied code must be on this allowlist:

| Allowed for copied code | Notes |
| --- | --- |
| `MPL-2.0` | Keep the license file and notices. |
| `Apache-2.0` | Preserve `NOTICE` when present. |
| `Apache-2.0 WITH LLVM-exception` | Preserve the exception text. |
| `BSD-2-Clause`, `BSD-3-Clause` | Preserve copyright and conditions. |
| `MIT`, `ISC`, `Zlib`, `BSL-1.0` | Preserve copyright and permission text. |
| `CC0-1.0`, public domain | Require documented evidence. |

These are not allowed for copied code: `GPL-*`, `LGPL-*`, `AGPL-*`, `SSPL-*`,
`EUPL-*` where incompatible, source-available licenses that restrict
commercial use, and code with no license. If a proposed addition needs one of
them, stop and escalate to the maintainer. Do not add it under a different
license.

License compatibility is checked against the intended use. A permissive or
MPL-2.0 component can be vendored with its own license intact. It must not be
relicensed as graphFlow code.

### 4. Preserve the text

Vendor under `include/graphFlow/thirdparty/<name>/`. Keep the upstream
directory shape and file contents. Keep the upstream license and notice files
in the same directory. Add a `README.md` that states:

- upstream URL and pinned SHA;
- SPDX identifier;
- "unmodified" or an exact list of local changes;
- the graphFlow change that introduced the copy.

Do not add graphFlow SPDX headers to upstream files. Their own headers stay
untouched. Add the upstream license to `LICENSES/` when REUSE needs it.

### 5. Update the notices

Add or update the entry in `THIRD_PARTY_NOTICES.md` and in the inventory above
in the same change. A vendoring change without a notice is incomplete.

### 6. Scan for copied text

Before review, check that no upstream text slipped into project-owned files.

```sh
# No vendored component may exist without an inventory entry.
find include/graphFlow/thirdparty -mindepth 1 -maxdepth 1 ! -name README.md

# Source, examples, tests, and build files must not name the copyleft
# references. Provenance pages are excluded here on purpose.
git grep -n -i -E 'spearhed|spmacc|picongpu' -- \
  include examples tests tools CMakeLists.txt cmake

# License metadata must be complete.
reuse lint
```

A match in the first command must have a matching inventory row. The second
command excludes the provenance pages because they name the references by
design. A name match in source code, a build file, or a test is a defect until
proven otherwise. A comment that credits a concept reference is reviewed for
copied expression as well.

For a new vendored component, also compare the copied tree with the pinned
upstream commit, for example with `git diff --no-index` against a clean
checkout, and confirm that only the documented metadata files differ.

### 7. Decide

Merge only when every step has evidence in the pull request. Otherwise remove
the material, replace it with an independent implementation, or escalate. The
audit is a merge gate, not a formality.

## Review checklist

- [ ] License is on the allowlist.
- [ ] Upstream URL and full commit SHA are recorded.
- [ ] License file is present and its SHA-256 is recorded.
- [ ] Upstream files are unmodified, or every change is documented.
- [ ] `THIRD_PARTY_NOTICES.md` and the inventory are updated.
- [ ] `reuse lint` passes.
- [ ] The copied-text scan is clean or explained.
- [ ] No GPL, LGPL, AGPL, SSPL, or unlicensed source was copied.
