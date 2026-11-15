<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Downstream patches to vendored llamaLite

The vendored tree is an unmodified copy of upstream except for the changes
listed here. Each note names the upstream commit the copy came from, the
compiler or platform that needs the change, and the condition for dropping the
patch once upstream carries an equivalent fix.

Upstream: <https://github.com/ComputationalRadiationPhysics/spearhed>, path
`packages/pmacc/libs/llamaLite`. Patches are written so they can be offered
upstream unchanged.

| Patch | Affected file | Reason |
| --- | --- | --- |
| [0001](0001-nvcc-12.5-explicit-lambda-template.md) | `include/llamaLite/Algorithm.hpp` | nvcc 12.5 rejects explicit-template-parameter lambdas inside a function template |
| [0002](0002-clang-unused-astype-parameter.md) | `include/llamaLite/traits.hpp` | clang with `-Wextra -Werror` rejects the unused `AsType` parameter name |
