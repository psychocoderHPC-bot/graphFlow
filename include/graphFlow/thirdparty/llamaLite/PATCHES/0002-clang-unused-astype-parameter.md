<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# 0002: unnamed `AsType` parameter for clang `-Wextra -Werror`

## Upstream

- Project: spearhed
- URL: <https://github.com/ComputationalRadiationPhysics/spearhed>
- Path: `packages/pmacc/libs/llamaLite/include/llamaLite/traits.hpp`
- Base commit: `aa79f32653452bdaea55d9786c57d1ab10a37b81`
- Upstream issue: not filed yet. Report it at
  <https://github.com/ComputationalRadiationPhysics/spearhed/issues>.

## Symptom

The HIP (ROCm 7.2, clang 22) and SYCL (oneAPI 2026.2, clang 22) builds enable
alpaka's pedantic flags (`-Wall -Wextra -Werror`) and fail on the template
definition of `llama_lite::traits::AsType::operator()`:

```text
traits.hpp:35:40: error: unused parameter 'fieldView' [-Werror,-Wunused-parameter]
        constexpr type operator()(auto fieldView) const
```

clang diagnoses the unused parameter when it parses the template definition.
The function body is empty by design: `AsType` is a tag whose only purpose is
the `_default_sentinel` check in `IsTraitSpecialized`.

## Patch

Comment out the parameter name:

```cpp
constexpr type operator()(auto /*fieldView*/) const
{
}
```

The signature, return type, callability, and template argument deduction are
unchanged.

## Verification

- graphFlow HIP and SYCL builds with `-Dalpaka_COMPILE_PEDANTIC=ON`.

## Removal

Drop this patch when the vendored copy moves to an upstream commit that marks
the parameter unused, or when the pedantic profiles no longer include
`-Wunused-parameter`.
