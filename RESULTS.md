# Benchmark Results

## 64-chip TPU v6e Pod

Sanitized D3Q27 SoA boilerplate, `1024 x 1024 x 1000`, 10 timed steps:

```json
{
  "devices": 64,
  "hosts": 16,
  "step_seconds": 0.6536441439999976,
  "glups": 1.604200098211243,
  "max_velocity": 0.020015714690089226,
  "rho_min": 0.9869076013565063,
  "rho_max": 1.0242279767990112,
  "passed": true
}
```

Explicit `jax.lax.scan` smoke test, `128 x 128 x 128`, 2 timed steps:

```json
{
  "step_seconds": 0.0030408234999868,
  "glups": 0.6896658092813028,
  "passed": true
}
```

## Vmap/SRAM-Fusion Run

Google suggested avoiding the local full-AoS materialization during collision.
The repo now includes `--collision-mode voxel_vmap`, which computes
hydrodynamic moments lazily from the 9 SoA groups and uses nested `jax.vmap`
around a one-voxel Cumulant micro-kernel.

The 1B benchmark completed successfully on the 64-chip v6e pod:

```json
{
  "devices": 64,
  "hosts": 16,
  "global_shape": [1024, 1024, 1000],
  "timed_steps": 10,
  "step_seconds": 0.6438548762002029,
  "glups": 1.6285906013297808,
  "max_velocity": 0.020015714690089226,
  "rho_min": 0.9869076013565063,
  "rho_max": 1.0242279767990112,
  "passed": true
}
```

Compared with the previous `local_aos`/full-local-27-channel run
(`1.604200098 GLUPS`), this is a modest but positive improvement of about
`1.52%`.

## Process-0 JAX Profiler Traces

The `voxel_vmap` path was profiled with `jax.profiler` using process-0 trace
capture. This keeps the multi-host run stable while still giving Google a real
XLA timeline for the SoA/vmap kernel on one host.

`512 x 512 x 500`, 3 timed steps:

```json
{
  "devices": 64,
  "hosts": 16,
  "global_shape": [512, 512, 500],
  "timed_steps": 3,
  "step_seconds": 0.07722084233440303,
  "glups": 1.697365582110537,
  "collision_mode": "voxel_vmap",
  "profile_processes": "process0",
  "max_velocity": 0.019992828369140625,
  "rho_min": 0.9905827641487122,
  "rho_max": 1.0179578065872192,
  "passed": true
}
```

`1024 x 1024 x 1000`, 3 timed steps:

```json
{
  "devices": 64,
  "hosts": 16,
  "global_shape": [1024, 1024, 1000],
  "timed_steps": 3,
  "step_seconds": 0.6443396653339732,
  "glups": 1.627365280168657,
  "collision_mode": "voxel_vmap",
  "profile_processes": "process0",
  "max_velocity": 0.019992828369140625,
  "rho_min": 0.9896554946899414,
  "rho_max": 1.0179983377456665,
  "passed": true
}
```

Trace artifacts are checked in under:

- `results/profiles/profile_voxel_vmap_512x512x500_3step/`
- `results/profiles/profile_voxel_vmap_1024x1024x1000_3step/`

The profiler warning `Can't import tensorflow.python.profiler.trace` appeared
on the pod, but the `.xplane.pb` and `.trace.json.gz` files were still written.
The legacy `local_aos` comparator was not used for profiler capture because it
stalled under the current TPU/XLA lowering even at small grids; the stable path
for the Google handoff is `--collision-mode voxel_vmap`.
