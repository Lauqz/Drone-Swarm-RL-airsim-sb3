# What is patched, and why

Every vendored package here was diffed against the exact version pinned in
`../environment.yml`. Only the three packages below actually differ; the rest
moved to `../archive/unmodified_libs/`.

Install only these three:

```bash
pip install -e modified_libs/supersuit
pip install -e modified_libs/stable_baselines3
pip install -e modified_libs/msgpackrpc
```

---

## supersuit 3.8.0

Five files differ. Two are general fixes that have been submitted upstream; three
are local workarounds specific to this project.

### Submitted upstream

| file | change | needed for | upstream |
|---|---|---|---|
| `multiagent_wrappers/black_death.py` | `black_death_v3` asserted every observation space is a `Box`, because it builds a zero observation with `np.zeros_like(space.low)` | **the default `train.py` chain** | [SuperSuit#269](https://github.com/Farama-Foundation/SuperSuit/pull/269) |
| `utils/frame_stack.py`, `generic_wrappers/frame_stack.py` | `frame_stack_v1/v2` rejected `Dict` observation spaces, so a camera-plus-position observation could not be stacked without flattening it first | only if the `frame_stack_v2` line in `scripts/airsim_env.py` is uncommented, which it is not by default | [SuperSuit#268](https://github.com/Farama-Foundation/SuperSuit/pull/268) |

The upstream versions are recursive rather than copy-pasted, so `Dict` reuses the
`Box` code path and nested `Dict`s work too. They were checked against the
implementation here and produce identical results on every case this one
supports. Once both merge, `supersuit` can be dropped from this folder.

Which patch is actually needed, measured against stock SuperSuit 3.11:

| SuperSuit | default `train.py` chain | with `frame_stack_v2` |
|---|---|---|
| unpatched | fails | fails |
| #269 only | works | fails |
| #268 only | fails | fails |
| both | works | works |

Unpatched, the run stops at `black_death_v3` with
`observation spaces for black death must be Box spaces`.

### Checked against current versions

With both patches the env passes `pettingzoo.test.parallel_api_test` on PettingZoo
1.26.1, and the full chain
(`black_death_v3` -> `pettingzoo_env_to_vec_env_v1` -> `concat_vec_envs_v1`) runs
on SuperSuit 3.11, NumPy 2.x and Gymnasium 1.3, giving
`{'cam': (3, 84, 84, 9), 'pos': (3, 9)}` with `frame_stack_v2(3)`.

Two source fixes were needed and are already applied: `np.fromstring` ->
`np.frombuffer().copy()` (binary mode removed in NumPy 2.0) and an explicit
`self.render_mode = None` (recent SuperSuit reads it).

These checks used a mocked AirSim client, so they cover the PettingZoo/SuperSuit
API surface rather than a full training run against the simulator.

### Local workarounds, not upstreamable

| file | change | note |
|---|---|---|
| `vector/markov_vector_wrapper.py` | forces `black_death=True` and casts observations per key, with `'cam'` hardcoded | project-specific. **Contains a bug**: `from ast import Dict` imports an AST node class, not `gymnasium.spaces.Dict`, so that `isinstance` check never matches what was intended |
| `vector/vector_constructors.py` | hardcodes `black_death=True` | project-specific default |
| `multiagent_wrappers/black_death.py` | also comments out `_check_valid_for_black_death()` | superseded by SuperSuit#269, which makes the check `Dict`-aware instead of removing it |

## stable_baselines3 2.0.0

Three files. SB3 2.0.0's `VecEnv` expected a 4-tuple `step_wait()` and a bare
`reset()`, while the env produced by `pettingzoo_env_to_vec_env_v1` returns a
5-tuple and `(obs, info)`.

- `common/vec_env/vec_transpose.py` — unpack `observations, rewards, dones, trunc, infos`; unpack `(obs, info)` from `reset()`
- `common/vec_env/base_vec_env.py` — same shape change
- `common/evaluation.py` — same, plus debug printing

Worth re-testing against a current SB3 before keeping this: the API mismatch it
works around may no longer exist.

## msgpackrpc 0.4.1

`client.py`: default RPC timeout raised from 10 s to 100 s. AirSim calls such as
`moveByVelocityBodyFrameAsync(...).join()` exceed 10 s under load and the run
dies with a timeout. One-line change, no upstream value.

---

## Verified unmodified

Moved to `../archive/unmodified_libs/`, byte-identical to the pinned upstream
release (line endings normalised):

| package | pinned | result |
|---|---|---|
| `pettingzoo` | 1.23.1 | identical, 0 files differ |
| `msgpack` | 0.5.6 (`msgpack-python`) | identical |
| `gymnasium` | 0.28.1 | one file differs, but only two unused imports and two trailing `#!!!` comments. Functionally a no-op, confirmed by comparing the parsed AST |

Install these from PyPI at the pinned versions instead.
