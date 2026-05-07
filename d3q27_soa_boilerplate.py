#!/usr/bin/env python3
"""Google-facing D3Q27 JAX SoA boilerplate.

This file is intentionally infrastructure-only.  It demonstrates the memory
layout, sharding strategy, graph decomposition, and branchless masking used by
the data-generation engine without embedding customer-specific geometry,
metadata, or domain-specific physics wrappers.

State layout:

    populations = tuple(9 arrays shaped (X, Y, Z, 3))

The resident distributed state is Struct-of-Arrays (SoA).  Each array is
sharded over X with ``PartitionSpec("x", None, None, None)``.  The default
collision path keeps the network state as 9 groups and uses nested ``jax.vmap``
to build the 27-channel vector only at the per-voxel collision boundary, giving
XLA the opportunity to fuse the local concatenate/collision/split inside TPU
SRAM tiles instead of materializing a full ``(local_X, Y, Z, 27)`` tensor in
HBM.  Boundary/solid handling uses ``jnp.where`` masks.

Note for infrastructure review:

    The production Cumulant collision module is represented here by the
    ``ShiftedD3Q27CumulantOperator`` import.  That module can be omitted from
    the shared boilerplate for IP reasons.  If Google needs to physically run
    this file for Parallelstore or HLO compilation experiments, replace that
    import with a dummy BGK collision operator exposing the same
    ``equilibrium_storage``, ``macroscopic``, and ``collide`` methods.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from shifted_cumulant_lbm import ShiftedD3Q27CumulantOperator
except ImportError as exc:  # pragma: no cover - this is an infra handoff guard.
    raise SystemExit(
        "Missing shifted_cumulant_lbm.py. The proprietary Cumulant collision "
        "module is intentionally not embedded in this boilerplate. For a "
        "runnable Google infra test, swap in a dummy BGK operator with the same "
        "equilibrium_storage/macroscopic/collide interface."
    ) from exc


def d3q27_velocities() -> np.ndarray:
    return np.asarray(
        [(cx, cy, cz) for cx in (-1, 0, 1) for cy in (-1, 0, 1) for cz in (-1, 0, 1)],
        dtype=np.int32,
    )


def d3q27_opposites(c: np.ndarray) -> np.ndarray:
    return np.asarray([np.where((c == -ci).all(axis=1))[0][0] for ci in c], dtype=np.int32)


def group_bounds(group_size: int = 3) -> list[tuple[int, int]]:
    return [(start, min(start + group_size, 27)) for start in range(0, 27, group_size)]


def split_groups(f: jnp.ndarray, bounds: list[tuple[int, int]]) -> tuple[jnp.ndarray, ...]:
    return tuple(f[..., start:stop].astype(jnp.float32) for start, stop in bounds)


def sync(name: str, enabled: bool) -> None:
    if enabled:
        multihost_utils.sync_global_devices(name)


def strain_rate_tensor(u: jnp.ndarray) -> jnp.ndarray:
    grads = [0.5 * (jnp.roll(u, -1, axis=axis) - jnp.roll(u, 1, axis=axis)) for axis in range(3)]
    grad = jnp.stack(grads, axis=-1)
    return 0.5 * (grad + jnp.swapaxes(grad, -1, -2))


def branchless_sgs_omega(
    u: jnp.ndarray,
    solid: jnp.ndarray,
    base_tau: float,
    smagorinsky_constant: float,
    tau_min: float = 0.5001,
    tau_max: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local relaxation rate using branchless solid masking."""
    strain = strain_rate_tensor(u)
    strain_mag = jnp.sqrt(2.0 * jnp.sum(strain * strain, axis=(-1, -2)) + 1e-30)
    nu0 = jnp.maximum((jnp.asarray(base_tau, dtype=u.dtype) - 0.5) / 3.0, 0.0)
    nu_sgs = (jnp.asarray(smagorinsky_constant, dtype=u.dtype) ** 2) * strain_mag
    tau_eff = jnp.clip(0.5 + 3.0 * (nu0 + nu_sgs), tau_min, tau_max)
    omega = 1.0 / tau_eff
    omega_solid = jnp.full_like(omega, 1.0 / jnp.asarray(base_tau, dtype=u.dtype))
    return jnp.where(solid, omega_solid, omega).astype(u.dtype), strain_mag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-x", type=int, default=1024)
    parser.add_argument("--y", type=int, default=1024)
    parser.add_argument("--z", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--pressure-gradient", type=float, default=1e-6)
    parser.add_argument("--inlet-velocity", type=float, default=0.02)
    parser.add_argument("--outlet-rho", type=float, default=1.0)
    parser.add_argument("--smagorinsky-constant", type=float, default=0.16)
    parser.add_argument(
        "--collision-mode",
        choices=("voxel_vmap", "local_aos"),
        default="voxel_vmap",
        help=(
            "voxel_vmap keeps the 27-channel concatenate inside a per-voxel "
            "vmap collision kernel so XLA can fuse it in SRAM tiles. "
            "local_aos preserves the older local full-AoS materialization path."
        ),
    )
    parser.add_argument(
        "--scan-chunk-steps",
        type=int,
        default=1,
        help=(
            "Number of LBM steps per jax.lax.scan chunk. Keep this small for "
            "graph decomposition; increase only when testing scan fusion."
        ),
    )
    parser.add_argument("--distributed-init", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional local or GCS TensorBoard trace directory for jax.profiler.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Optional profiler label stored in the JSON report.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/google_infra/d3q27_soa_boilerplate.json"))
    args = parser.parse_args()

    if args.distributed_init:
        jax.distributed.initialize()

    devices = jax.devices()
    ndev = len(devices)
    if args.global_x % ndev:
        raise SystemExit(f"--global-x={args.global_x} must be divisible by device_count={ndev}")

    mesh = Mesh(np.asarray(devices), ("x",))
    bounds = group_bounds(3)
    velocities = d3q27_velocities()
    velocities_jnp = jnp.asarray(velocities, dtype=jnp.float32)
    opposites = d3q27_opposites(velocities)
    op = ShiftedD3Q27CumulantOperator(tau=args.tau, rho0=1.0, dtype=jnp.float32)

    sharding = NamedSharding(mesh, P("x", None, None, None))
    local_x = args.global_x // ndev

    def make_zero_group() -> jax.Array:
        def callback(index):
            x_slice, y_slice, z_slice, q_slice = index
            shape = (
                len(range(*x_slice.indices(args.global_x))),
                len(range(*y_slice.indices(args.y))),
                len(range(*z_slice.indices(args.z))),
                len(range(*q_slice.indices(3))),
            )
            return np.zeros(shape, dtype=np.float32)

        return jax.make_array_from_callback((args.global_x, args.y, args.z, 3), sharding, callback)

    groups = tuple(make_zero_group() for _ in bounds)
    in_specs = tuple(P("x", None, None, None) for _ in bounds)
    out_specs = in_specs

    def generic_solid_mask(local_shape: tuple[int, ...], x_offset: jnp.ndarray) -> jnp.ndarray:
        """Procedural generic mask: cylinder wall plus four simple baffles."""
        x = (x_offset + jnp.arange(local_shape[0], dtype=jnp.float32))[:, None, None]
        y = jnp.arange(args.y, dtype=jnp.float32)[None, :, None]
        z = jnp.arange(args.z, dtype=jnp.float32)[None, None, :]
        cy = (args.y - 1.0) / 2.0
        cz = (args.z - 1.0) / 2.0
        radius = 0.46 * min(args.y, args.z)
        r = jnp.sqrt((y - cy) ** 2 + (z - cz) ** 2)

        outer_wall = r > radius
        baffle_width = 0.035 * radius
        near_wall = r > 0.70 * radius
        baffle_y = near_wall & (jnp.abs(y - cy) < baffle_width)
        baffle_z = near_wall & (jnp.abs(z - cz) < baffle_width)
        short_generic_insert = (x > 0.35 * args.global_x) & (x < 0.39 * args.global_x) & near_wall & (y > cy) & (z > cz)
        return outer_wall | baffle_y | baffle_z | short_generic_insert

    def init_local(*local_groups):
        shard_idx = jax.lax.axis_index("x")
        x_offset = shard_idx * local_groups[0].shape[0]
        solid = generic_solid_mask(local_groups[0].shape, x_offset)

        y = jnp.arange(args.y, dtype=jnp.float32)[None, :, None]
        z = jnp.arange(args.z, dtype=jnp.float32)[None, None, :]
        cy = (args.y - 1.0) / 2.0
        cz = (args.z - 1.0) / 2.0
        rr = ((y - cy) / (0.46 * min(args.y, args.z))) ** 2 + ((z - cz) / (0.46 * min(args.y, args.z))) ** 2
        profile = jnp.maximum(0.0, 1.0 - rr)

        rho = jnp.ones((local_groups[0].shape[0], args.y, args.z), dtype=jnp.float32)
        u = jnp.zeros((local_groups[0].shape[0], args.y, args.z, 3), dtype=jnp.float32)
        u = u.at[..., 0].set(0.25 * args.inlet_velocity * profile)
        u = jnp.where(solid[..., None], 0.0, u)
        f_eq = op.equilibrium_storage(rho, u)
        f_eq = jnp.where(solid[..., None], 0.0, f_eq)
        return split_groups(f_eq, bounds)

    initialize = jax.jit(shard_map(init_local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False))
    groups = initialize(*groups)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), groups)
    sync("d3q27_soa_boilerplate_init", args.distributed_init)

    send_right = [(i, (i + 1) % ndev) for i in range(ndev)]
    send_left = [(i, (i - 1) % ndev) for i in range(ndev)]
    group_of = {q: gi for gi, (start, stop) in enumerate(bounds) for q in range(start, stop)}
    local_of = {q: q - bounds[group_of[q]][0] for q in range(27)}

    def macro_from_groups(local_groups):
        """Hydrodynamic moments without materializing the 27-population tensor."""
        rho_delta = jnp.zeros(local_groups[0].shape[:-1], dtype=local_groups[0].dtype)
        momentum = jnp.zeros(local_groups[0].shape[:-1] + (3,), dtype=local_groups[0].dtype)
        for group, (start, stop) in zip(local_groups, bounds):
            rho_delta = rho_delta + jnp.sum(group, axis=-1)
            momentum = momentum + jnp.einsum("...q,qd->...d", group, velocities_jnp[start:stop])
        rho = jnp.asarray(op.rho0, dtype=local_groups[0].dtype) + rho_delta
        u = momentum / rho[..., None]
        return rho.astype(jnp.float32), u.astype(jnp.float32)

    def collide_micro_voxel(*items):
        """One-voxel collision used by nested vmap.

        The only 27-wide tensor here is a single voxel vector.  That is the
        shape Google asked for: local SoA fields enter separately, the Cumulant
        math sees a compact vector inside the mapped kernel, and the result is
        split back before it can become a full local AoS allocation.
        """
        group_vecs = items[:9]
        rho_scalar, u_vec, omega_scalar, solid_scalar = items[9:]
        f_vec = jnp.concatenate(group_vecs, axis=-1)
        f = f_vec.reshape((1, 1, 1, 27))
        rho = rho_scalar.reshape((1, 1, 1))
        u = u_vec.reshape((1, 1, 1, 3))
        omega = omega_scalar.reshape((1, 1, 1))

        f_post = op.collide(f, omega_shear=omega)
        du = jnp.zeros_like(u).at[..., 0].set(args.pressure_gradient)
        f_forced = f_post + (op.equilibrium_storage(rho, u + du) - op.equilibrium_storage(rho, u))
        f_forced = jnp.where(solid_scalar, 0.0, f_forced.reshape((27,)))
        return tuple(f_forced[start:stop].astype(jnp.float32) for start, stop in bounds)

    vmapped_collide_micro = jax.vmap(
        jax.vmap(
            jax.vmap(
                collide_micro_voxel,
                in_axes=(*([0] * 9), 0, 0, 0, 0),
                out_axes=0,
            ),
            in_axes=(*([0] * 9), 0, 0, 0, 0),
            out_axes=0,
        ),
        in_axes=(*([0] * 9), 0, 0, 0, 0),
        out_axes=0,
    )

    def collide_local(*local_groups):
        shard_idx = jax.lax.axis_index("x")
        x_offset = shard_idx * local_groups[0].shape[0]
        solid = generic_solid_mask(local_groups[0].shape, x_offset)

        rho, u = macro_from_groups(local_groups)
        u = jnp.where(solid[..., None], 0.0, u)

        omega, strain_mag = branchless_sgs_omega(u, solid, args.tau, args.smagorinsky_constant)
        if args.collision_mode == "local_aos":
            # Legacy comparator: materialize a local (local_X, Y, Z, 27) tensor.
            f = jnp.concatenate(local_groups, axis=-1)
            f_post = jax.checkpoint(lambda f_in, omega_in: op.collide(f_in, omega_shear=omega_in), prevent_cse=False)(
                f, omega
            )
            du = jnp.zeros_like(u).at[..., 0].set(args.pressure_gradient)
            f_forced = f_post + (op.equilibrium_storage(rho, u + du) - op.equilibrium_storage(rho, u))
            out_groups = split_groups(jnp.where(solid[..., None], 0.0, f_forced), bounds)
        else:
            out_groups = vmapped_collide_micro(*local_groups, rho, u, omega, solid)
        return (*out_groups, jax.lax.pmax(jnp.max(strain_mag), "x"))

    collide = jax.jit(
        shard_map(
            collide_local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(*out_specs, P()),
            check_rep=False,
        )
    )

    def make_stream_group(start: int, stop: int):
        channel_indices = list(range(start, stop))

        def stream_local(*local_groups):
            shard_idx = jax.lax.axis_index("x")
            x_offset = shard_idx * local_groups[0].shape[0]
            solid_here = generic_solid_mask(local_groups[0].shape, x_offset)
            local_group = local_groups[group_of[start]]

            left = jax.lax.ppermute(local_group[-1:, :, :, :], axis_name="x", perm=send_right)
            right = jax.lax.ppermute(local_group[:1, :, :, :], axis_name="x", perm=send_left)
            expanded = jnp.concatenate([left, local_group, right], axis=0)

            chunks = []
            for local_i, global_i in enumerate(channel_indices):
                cx, cy, cz = velocities[global_i].tolist()
                opp = int(opposites[global_i])

                pulled = expanded[1 - int(cx) : 1 - int(cx) + local_group.shape[0], :, :, local_i]
                pulled = jnp.roll(pulled, shift=int(cy), axis=1)
                pulled = jnp.roll(pulled, shift=int(cz), axis=2)

                neighbor_solid = generic_solid_mask(local_groups[0].shape, x_offset - int(cx))
                neighbor_solid = jnp.roll(neighbor_solid, shift=-int(cy), axis=1)
                neighbor_solid = jnp.roll(neighbor_solid, shift=-int(cz), axis=2)
                boundary = (~solid_here) & neighbor_solid
                bounce = local_groups[group_of[opp]][..., local_of[opp]]

                # Branchless mask: select bounce-back at walls, regular stream elsewhere.
                value = jnp.where(boundary, bounce, pulled)
                value = jnp.where(solid_here, 0.0, value)
                chunks.append(value[..., None])
            return jnp.concatenate(chunks, axis=-1)

        return jax.jit(
            shard_map(
                stream_local,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=P("x", None, None, None),
                check_rep=False,
            )
        )

    streamers = tuple(make_stream_group(start, stop) for start, stop in bounds)

    def regularized_plane(interior_f: jnp.ndarray, target_rho: jnp.ndarray, target_u: jnp.ndarray) -> jnp.ndarray:
        rho_i, u_i = op.macroscopic(interior_f)
        return (op.equilibrium_storage(target_rho, target_u) + interior_f - op.equilibrium_storage(rho_i, u_i)).astype(jnp.float32)

    def open_x_local(*local_groups):
        shard_idx = jax.lax.axis_index("x")
        f = jnp.concatenate(local_groups, axis=-1)
        solid = generic_solid_mask(local_groups[0].shape, shard_idx * local_groups[0].shape[0])

        def apply_inlet(local_state):
            y = jnp.arange(args.y, dtype=local_state.dtype)[None, :, None]
            z = jnp.arange(args.z, dtype=local_state.dtype)[None, None, :]
            cy = (args.y - 1.0) / 2.0
            cz = (args.z - 1.0) / 2.0
            rr = ((y - cy) / (0.46 * min(args.y, args.z))) ** 2 + ((z - cz) / (0.46 * min(args.y, args.z))) ** 2
            profile = jnp.maximum(0.0, 1.0 - rr)
            inlet_rho = jnp.full((1, args.y, args.z), 1.0, dtype=local_state.dtype)
            inlet_u = jnp.zeros((1, args.y, args.z, 3), dtype=local_state.dtype).at[..., 0].set(args.inlet_velocity * profile)
            plane = regularized_plane(local_state[1:2, :, :, :], inlet_rho, inlet_u)
            plane = jnp.where(solid[0:1, :, :, None], local_state[0:1, :, :, :], plane)
            return local_state.at[0:1, :, :, :].set(plane)

        def apply_outlet(local_state):
            rho_o, u_o = op.macroscopic(local_state[-2:-1, :, :, :])
            outlet_rho = jnp.full_like(rho_o, args.outlet_rho)
            outlet_u = u_o.at[..., 1].set(0.0).at[..., 2].set(0.0)
            outlet_u = outlet_u.at[..., 0].set(jnp.maximum(outlet_u[..., 0], 0.0))
            plane = regularized_plane(local_state[-2:-1, :, :, :], outlet_rho, outlet_u)
            plane = jnp.where(solid[-1:, :, :, None], local_state[-1:, :, :, :], plane)
            return local_state.at[-1:, :, :, :].set(plane)

        f = jax.lax.cond(shard_idx == 0, apply_inlet, lambda x: x, f)
        f = jax.lax.cond(shard_idx == ndev - 1, apply_outlet, lambda x: x, f)
        f = jnp.where(solid[..., None], 0.0, f)
        return split_groups(f, bounds)

    open_x = jax.jit(shard_map(open_x_local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False))

    def metrics_local(*local_groups):
        f = jnp.concatenate(local_groups, axis=-1)
        rho, u = op.macroscopic(f)
        speed = jnp.linalg.norm(u, axis=-1)
        finite = jnp.all(jnp.isfinite(f)) & jnp.all(jnp.isfinite(rho)) & jnp.all(jnp.isfinite(u))
        return (
            jax.lax.pmax(jnp.max(speed), "x"),
            jax.lax.pmin(jnp.min(rho), "x"),
            jax.lax.pmax(jnp.max(rho), "x"),
            jax.lax.pmin(finite.astype(jnp.int32), "x"),
        )

    metrics = jax.jit(shard_map(metrics_local, mesh=mesh, in_specs=in_specs, out_specs=(P(), P(), P(), P()), check_rep=False))

    def one_step(groups):
        result = collide(*groups)
        groups = result[:9]
        aux = result[9]
        groups = tuple(streamer(*groups) for streamer in streamers)
        groups = open_x(*groups)
        return groups, aux

    def scan_body(groups, _):
        # The time loop is explicit JAX control flow.  The heavy per-step work
        # still goes through the shard_map kernels above:
        #
        #   collide_local: SoA -> jnp.concatenate(..., axis=-1) -> local AoS
        #   stream_local: per-group ppermute halo exchange over the TPU ICI
        #   open_x_local: regularized shard-edge inlet/outlet
        return one_step(groups)

    def run_scan_chunk(groups, steps: int):
        groups, aux_series = jax.lax.scan(scan_body, groups, xs=None, length=steps)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), groups)
        aux = aux_series[-1] if steps else jnp.asarray(0.0, dtype=jnp.float32)
        return groups, aux

    if args.scan_chunk_steps < 1:
        raise SystemExit("--scan-chunk-steps must be >= 1")

    remaining = args.warmup_steps
    strain_max = jnp.asarray(0.0, dtype=jnp.float32)
    while remaining > 0:
        chunk = min(args.scan_chunk_steps, remaining)
        groups, strain_max = run_scan_chunk(groups, chunk)
        remaining -= chunk

    sync("d3q27_soa_boilerplate_timed_start", args.distributed_init)
    if args.profile_dir is not None:
        profile_dir = str(args.profile_dir)
        if not profile_dir.startswith("gs://") and jax.process_index() == 0:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
        sync("d3q27_soa_boilerplate_profile_dir_ready", args.distributed_init)
        jax.profiler.start_trace(profile_dir)

    t0 = time.perf_counter()
    remaining = args.timed_steps
    while remaining > 0:
        chunk = min(args.scan_chunk_steps, remaining)
        groups, strain_max = run_scan_chunk(groups, chunk)
        remaining -= chunk
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), groups)
    elapsed = time.perf_counter() - t0

    if args.profile_dir is not None:
        jax.profiler.stop_trace()
    sync("d3q27_soa_boilerplate_timed_done", args.distributed_init)

    max_speed, rho_min, rho_max, finite_i32 = jax.device_get(metrics(*groups))
    cells = args.global_x * args.y * args.z
    report = {
        "test": "google_infra_d3q27_soa_boilerplate",
        "layout": "(X,Y,Z,3) x 9 SoA",
        "devices": ndev,
        "hosts": jax.process_count(),
        "global_shape": [args.global_x, args.y, args.z],
        "logical_population_shape": [args.global_x, args.y, args.z, 27],
        "local_x_per_device": local_x,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "timed_seconds": elapsed,
        "step_seconds": elapsed / max(args.timed_steps, 1),
        "glups": cells * args.timed_steps / max(elapsed, 1e-12) / 1e9,
        "collision_mode": args.collision_mode,
        "profile_dir": str(args.profile_dir) if args.profile_dir is not None else None,
        "profile_name": args.profile_name,
        "max_velocity": float(np.asarray(max_speed).reshape(-1)[0]),
        "rho_min": float(np.asarray(rho_min).reshape(-1)[0]),
        "rho_max": float(np.asarray(rho_max).reshape(-1)[0]),
        "strain_max": float(np.asarray(jax.device_get(strain_max)).reshape(-1)[0]),
        "finite": bool(np.asarray(finite_i32).reshape(-1)[0]),
        "features": {
            "soa_groups": 9,
            "channels_per_group": 3,
            "network_soa_compute_aos": True,
            "branchless_wall_masking": True,
            "regularized_open_x": True,
            "checkpointed_collision": True,
            "voxel_vmap_collision": args.collision_mode == "voxel_vmap",
        },
    }
    report["passed"] = bool(report["finite"] and report["max_velocity"] < 0.18 and 0.95 <= report["rho_min"] <= report["rho_max"] <= 1.05)

    if jax.process_index() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
