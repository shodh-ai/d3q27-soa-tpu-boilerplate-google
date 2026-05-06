# D3Q27 SoA Boilerplate Reproduction Bundle

This bundle is sanitized for infrastructure review.

## Files

- `d3q27_soa_boilerplate.py`: JAX D3Q27 solver boilerplate using `(X,Y,Z,3) x 9` SoA.
- `requirements-tpu.txt`: TPU Python dependencies.
- `requirements-gpu.txt`: GPU Python dependencies.
- `launch_tpu_v6e64_d3q27_soa.sh`: 64-chip TPU pod launch command.
- `run_h100_d3q27_soa.sbatch`: one-node 8x H100 Slurm template.

## Collision Math Note

The production Cumulant collision implementation is represented by the
`ShiftedD3Q27CumulantOperator` import. That physics module can be omitted from
the external handoff for IP reasons. If Google needs to physically execute the
script to test Parallelstore I/O or HLO compilation behavior, replace the import
with a dummy BGK operator exposing:

- `equilibrium_storage(rho, u)`
- `macroscopic(delta_f)`
- `collide(delta_f, omega_shear=None)`

## Architecture Shown

- Resident state is Struct-of-Arrays: 9 arrays shaped `(X,Y,Z,3)`.
- X-axis sharding uses `PartitionSpec("x", None, None, None)`.
- Network transfer uses SoA groups for halo exchange.
- Local compute temporarily fuses groups with `jnp.concatenate(..., axis=-1)`.
- Solid and boundary masks use `jnp.where`.
- Collision is graph-split with `jax.checkpoint` and explicit `block_until_ready`.
- The time loop is written as `jax.lax.scan` chunks, controlled by
  `--scan-chunk-steps`. Default is `1` to preserve graph decomposition.

## TPU Setup

```bash
python3 -m pip install --user -r requirements-tpu.txt \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Copy the boilerplate and collision core to every TPU host:

```bash
gcloud alpha compute tpus tpu-vm scp --worker=all \
  d3q27_soa_boilerplate.py shifted_cumulant_lbm.py \
  ${TPU_NAME}:~/skanda_simulation/
```

Run:

```bash
PROJECT=YOUR_GCP_PROJECT \
ZONE=YOUR_TPU_ZONE \
TPU_NAME=YOUR_TPU_POD_NAME \
./launch_tpu_v6e64_d3q27_soa.sh
```

## H100 Slurm Setup

```bash
python -m pip install -r requirements-gpu.txt
sbatch run_h100_d3q27_soa.sbatch
```

Override size:

```bash
GLOBAL_X=1024 Y=1024 Z=1000 TIMED_STEPS=10 sbatch run_h100_d3q27_soa.sbatch
```

## Output Metric

The script writes JSON with:

- `glups`
- `step_seconds`
- `timed_seconds`
- `devices`
- `hosts`
- `global_shape`
- `layout`
- `max_velocity`
- `rho_min`
- `rho_max`
- `passed`

The sanitized boilerplate benchmark on the 64-chip v6e pod reached
`1.604200098 GLUPS` at `1024 x 1024 x 1000` for 10 timed steps:

```json
{
  "timed_seconds": 6.536441439999976,
  "step_seconds": 0.6536441439999976,
  "glups": 1.604200098211243,
  "passed": true
}
```

The explicit `jax.lax.scan` version was also smoke-tested on all 64 chips at
`128 x 128 x 128` for 2 timed steps:

```json
{
  "step_seconds": 0.0030408234999868,
  "glups": 0.6896658092813028,
  "passed": true
}
```
