# Unmodified vendored libraries

These were shipped alongside the patched ones when the original README said
"I can't remember which packages I modified". They were later diffed against the
exact versions pinned in `../../environment.yml` and turned out not to be
modified at all.

| package | pinned version | result |
|---|---|---|
| `pettingzoo` | 1.23.1 | byte-identical to upstream, 0 files differ |
| `msgpack` | 0.5.6 (`msgpack-python`) | byte-identical to upstream |
| `gymnasium` | 0.28.1 | one file differs (`vector/utils/numpy_utils.py`) but only by two unused imports and two trailing `#!!!` comments; the parsed AST is otherwise identical, so it is a no-op |

**Do not install from here.** Use the pinned releases:

```bash
pip install pettingzoo==1.23.1 msgpack-python==0.5.6 gymnasium==0.28.1
```

They are kept only so the original setup can be reproduced exactly. See
`../../modified_libs/PATCHES.md` for the packages that are genuinely patched.
