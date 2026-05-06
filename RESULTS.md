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
