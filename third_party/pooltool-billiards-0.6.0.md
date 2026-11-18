# pooltool-billiards 0.6.0 dependency metadata patch

- Upstream distribution: `pooltool-billiards==0.6.0`, published on PyPI on
  2026-03-16 under Apache-2.0 and requiring Python `>=3.10,<3.14`.
- Upstream wheel:
  `https://files.pythonhosted.org/packages/7c/14/bf4fac0e16abf8245399a2d0e14444c717722299684f6e319a0117baf8d8/pooltool_billiards-0.6.0-py3-none-any.whl`
- Verified upstream wheel SHA-256:
  `c47a46aeeeac6eea645e5c1610211ce0f0ae350e4b77909b8dd55b66e77b3bd5`.
- License evidence: wheel `METADATA` declares `License-Expression: Apache-2.0`
  and includes `pooltool_billiards-0.6.0.dist-info/licenses/LICENSE.txt`.

## Patch and reason

The upstream dependency metadata requires `panda3d==1.11.0.dev3702` outside
Darwin, but that build is absent from PyPI. This makes an otherwise universal
project lock unsatisfiable. The project therefore applies one controlled uv
override, `panda3d>=1.10.13,<1.11`, matching the stable Panda3D range already
declared by pooltool for Darwin. The resolved version is recorded in `uv.lock`.

The patch changes dependency metadata only. No pooltool physics algorithm or source file is modified,
copied, or vendored. The pooltool wheel remains the
verified upstream artifact above, and its hash remains pinned in `uv.lock`.
