<!-- SPDX-FileCopyrightText: 2026 René Widera -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# llamaLite upstream provenance

- Upstream project: spearhed
- Upstream URL: <https://github.com/ComputationalRadiationPhysics/spearhed>
- Upstream path: `packages/pmacc/libs/llamaLite`
- Upstream commit: `aa79f32653452bdaea55d9786c57d1ab10a37b81`
- SPDX identifier: `MPL-2.0`
- License file: `LICENSE`, SHA-256 `3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04`
- Modification state: unmodified. Every file listed below is byte-identical to the
  same path in the upstream commit. graphFlow adds only this provenance file.

The headers are exported by the `graphFlow::graphFlow` CMake target, so consumers
include the public API as `<llamaLite/llamaLite.hpp>`. The vendored tree is not
built as part of graphFlow; the upstream CMake files are kept unmodified so the
copy can be compared against upstream.

## Files (34)

```text
CMakeLists.txt
LICENSE
README.md
cmake/llamaLiteConfig.cmake.in
include/llamaLite/AccessSet.hpp
include/llamaLite/Algorithm.hpp
include/llamaLite/DynSoA.hpp
include/llamaLite/Field.hpp
include/llamaLite/One.hpp
include/llamaLite/Record.hpp
include/llamaLite/ResolveLeaf.hpp
include/llamaLite/Set.hpp
include/llamaLite/Soa.hpp
include/llamaLite/SubRecord.hpp
include/llamaLite/Transform.hpp
include/llamaLite/Tuple.hpp
include/llamaLite/View.hpp
include/llamaLite/llamaLite.hpp
include/llamaLite/tag/TagPath.hpp
include/llamaLite/tag/identifier.hpp
include/llamaLite/traits.hpp
include/llamaLite/utility.hpp
tests/CMakeLists.txt
tests/standalone/CMakeLists.txt
tests/standalone/main.cpp
tests/unit/accessSet/accessSet.cpp
tests/unit/dynSoa/dynSoa.cpp
tests/unit/iterate/iterate.cpp
tests/unit/one/one.cpp
tests/unit/record/record.cpp
tests/unit/set/set.cpp
tests/unit/soa/soa.cpp
tests/unit/tagPath/tagPath.cpp
tests/unit/view/view.cpp
```
