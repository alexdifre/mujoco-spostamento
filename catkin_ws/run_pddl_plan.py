#!/usr/bin/env python3
import argparse
import os
import sys
import time
from collections import deque

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from arm_dynamics import ArmDynamics
from acados_rti_solver import AcadosRTIConfig, AcadosRTISolver
from collision_spheres import (
    default_table_box_sdf,
    make_default_ur10e_collision_model,
)
from environment import (
    PDDL_CYLINDER_RADII,
    environment,
    load_pddl_bucket_targets,
)
from rti_sqp_mpc import (
    ArmNMPCProblem,
)
from viz import draw_polyline, draw_sphere_marker


DEFAULT_TARGET_CYLINDER = "out-2"

TRAJ_MAX_PTS = 800
TRAJ_SAMPLE_DT = 0.01


def default_acados_export_dir(name="ur10e_above"):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "acados_generated",
        name,
    ))


def resolve_acados_export_dir(path):
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        path,
    ))


def build_single_target(target_cylinder, target_clearance=0.06,
                        rotate_layout=True):
    if target_cylinder not in PDDL_CYLINDER_RADII:
        options = ", ".join(sorted(PDDL_CYLINDER_RADII))
        raise KeyError(
            f"unknown visible cylinder {target_cylinder!r}; choose one of: "
            f"{options}"
        )
    targets = load_pddl_bucket_targets(
        rotate_layout=rotate_layout,
        z_clearance=target_clearance,
    )
    if target_cylinder not in targets:
        raise KeyError(f"missing PDDL target for {target_cylinder!r}")
    return targets[target_cylinder], f"above {target_cylinder} center"


def parabolic_arc_height(p0, p_goal):
    p0 = np.asarray(p0, dtype=np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    distance = float(np.linalg.norm(p_goal - p0))
    return max(0.08, 0.35 * distance)


def build_parabolic_waypoints(p0, p_goal, num_waypoints=80):
    p0 = np.asarray(p0, dtype=np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    count = max(2, int(num_waypoints))

    midpoint = 0.5 * (p0 + p_goal)
    p_control = midpoint + np.array(
        [0.0, 0.0, parabolic_arc_height(p0, p_goal)],
        dtype=np.float64,
    )

    waypoints = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        s = i / float(count - 1)
        waypoints[i] = (
            (1.0 - s) ** 2 * p0
            + 2.0 * (1.0 - s) * s * p_control
            + s ** 2 * p_goal
        )
    return waypoints


def build_linear_waypoints(p0, p_goal, num_waypoints=24):
    p0 = np.asarray(p0, dtype=np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    count = max(2, int(num_waypoints))
    s_values = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return p0[None, :] + s_values[:, None] * (p_goal - p0)[None, :]


def waypoint_arclengths(waypoints):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return np.zeros(len(waypoints), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def sample_waypoint_path(waypoints, arclengths, distances):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    arclengths = np.asarray(arclengths, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    distances = np.clip(distances, 0.0, float(arclengths[-1]))
    samples = np.empty((distances.size, waypoints.shape[1]), dtype=np.float64)
    for axis in range(waypoints.shape[1]):
        samples[:, axis] = np.interp(distances, arclengths, waypoints[:, axis])
    return samples


def reference_horizon_distances(start_distance, horizon, dt, speed,
                                lookahead=0.0, total_distance=None,
                                ):
    dt = float(dt)
    distance = float(start_distance) + float(lookahead)
    distances = np.empty(horizon + 1, dtype=np.float64)
    if total_distance is None:
        step_distance = max(float(speed) * dt, 1e-6)
        return distance + step_distance * np.arange(horizon + 1)

    total_distance = float(total_distance)
    distance = min(distance, total_distance)
    for k in range(horizon + 1):
        distances[k] = distance
        remaining = total_distance - distance
        if remaining <= 0.0:
            continue
        step_distance = max(float(speed) * dt, 1e-6)
        distance = min(total_distance, distance + step_distance)
    return distances


def reference_horizon_from_path(waypoints, arclengths, start_distance,
                                horizon, dt, speed, lookahead=0.0):
    distances = reference_horizon_distances(
        start_distance,
        horizon,
        dt,
        speed,
        lookahead=lookahead,
        total_distance=float(arclengths[-1]),
    )
    return sample_waypoint_path(waypoints, arclengths, distances)


def reference_velocities_from_path(waypoints, arclengths, distances, speed):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    arclengths = np.asarray(arclengths, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    velocities = np.zeros((distances.size, waypoints.shape[1]),
                          dtype=np.float64)
    if len(waypoints) < 2 or float(speed) <= 0.0:
        return velocities

    segment_indices = np.searchsorted(
        arclengths,
        np.clip(distances, 0.0, float(arclengths[-1])),
        side="right",
    ) - 1
    segment_indices = np.clip(segment_indices, 0, len(waypoints) - 2)
    for i, segment_index in enumerate(segment_indices):
        ds = float(arclengths[segment_index + 1] - arclengths[segment_index])
        if ds > 1e-12:
            tangent = (waypoints[segment_index + 1]
                       - waypoints[segment_index]) / ds
            velocities[i] = float(speed) * tangent
    return velocities


def closest_waypoint_index(ee_pos, waypoints):
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    waypoints = np.asarray(waypoints, dtype=np.float64)
    distances = np.linalg.norm(waypoints - ee_pos[None, :], axis=1)
    return int(np.argmin(distances)), float(np.min(distances))


def update_path_progress(ee_pos, waypoints, arclengths, path_progress,
                         dt, speed, max_path_lead, tracking_tolerance):
    closest_index, distance_to_path = closest_waypoint_index(ee_pos, waypoints)
    closest_progress = float(arclengths[closest_index])
    lead_limit = closest_progress + float(max_path_lead)
    progress = min(max(float(path_progress), closest_progress), lead_limit)
    advance_radius = max(float(max_path_lead), float(tracking_tolerance), 1e-9)
    if distance_to_path <= advance_radius:
        progress += max(float(speed), 0.0) * float(dt)
    progress = min(progress, lead_limit)
    progress = min(progress, float(arclengths[-1]))
    progress_index = int(np.searchsorted(arclengths, progress, side="right") - 1)
    progress_index = int(np.clip(progress_index, 0, len(waypoints) - 1))
    return progress, progress_index, closest_index, distance_to_path


def limit_torque_slew(desired_tau, previous_tau, max_delta_tau):
    desired_tau = np.asarray(desired_tau, dtype=np.float64)
    previous_tau = np.asarray(previous_tau, dtype=np.float64)
    max_delta_tau = float(max_delta_tau)
    if max_delta_tau <= 0.0:
        return desired_tau
    return np.clip(
        desired_tau,
        previous_tau - max_delta_tau,
        previous_tau + max_delta_tau,
    )


def apply_optional_tau_limit(tau, apply_tau_limit):
    tau = np.asarray(tau, dtype=np.float64)
    limit = float(apply_tau_limit)
    if limit <= 0.0:
        return tau
    return np.clip(tau, -limit, limit)


def pace_realtime(robot, wall_start, sim_start, real_time_factor):
    factor = max(float(real_time_factor), 1e-9)
    sim_elapsed = float(robot.data.time) - float(sim_start)
    wall_elapsed = time.perf_counter() - float(wall_start)
    sleep_time = sim_elapsed / factor - wall_elapsed
    if sleep_time > 0.0:
        time.sleep(min(sleep_time, 0.02))


def terminal_upright_error(robot):
    return float(np.linalg.norm(robot.ee_rot[:2, 2]))


def phase_acados_export_dir(args, export_suffix=""):
    if export_suffix == "vertical":
        directory = args.acados_insert_export_dir
    else:
        directory = args.acados_above_export_dir
    if not args.collision_constraints:
        directory = f"{directory}_free"
    return resolve_acados_export_dir(directory)


def make_solver(args, env, initial_pos, target_pos, export_suffix=""):
    arm = ArmDynamics.from_robot(env.robot, dt=args.mpc_dt)
    initial_pos = np.asarray(initial_pos, dtype=np.float64)
    refs = np.repeat(initial_pos[None, :], args.horizon + 1, axis=0)
    collision_model = None
    if args.collision_constraints:
        box_sdf = default_table_box_sdf() if args.include_box else None
        excluded_obstacles = []
        if not args.avoid_target_cylinder:
            excluded_obstacles.append(
                f"pddl_{args.target_cylinder.replace('-', '_')}_cylinder")
        collision_model = make_default_ur10e_collision_model(
            env,
            arm,
            include_box=args.include_box,
            box_sdf=box_sdf,
            d_ground=args.d_ground,
            d_safe=args.d_safe,
            d_box=args.d_box,
            excluded_obstacle_names=excluded_obstacles,
        )
    problem = ArmNMPCProblem(
        arm,
        args.horizon,
        refs,
        Qp=[args.ee_pos_weight, args.ee_pos_weight, args.ee_z_weight],
        Qpv=[args.ee_velocity_weight] * 3,
        Qq=[args.q_weight] * 6,
        Qv=[args.qv_weight] * 6,
        Qf=[
            args.ee_terminal_weight,
            args.ee_terminal_weight,
            args.ee_terminal_z_weight,
        ],
        Qaxis=[
            args.ee_upright_weight,
            args.ee_upright_weight,
            0.0,
        ],
        Qaxisf=[
            args.ee_terminal_upright_weight,
            args.ee_terminal_upright_weight,
            0.0,
        ],
        Qqf=[args.qf_weight] * 6,
        Qvf=[args.qvf_weight] * 6,
        Rd=[args.delta_tau_cost] * 6,
        q_nominal=env.robot._home,
        previous_tau=np.zeros(6),
        collision_model=collision_model,
        terminal_axis=[0.0, 0.0, -1.0],
        terminal_axis_index=2,
        delta_q_max=[args.delta_q_max] * 6,
        delta_dq_max=[args.delta_dq_max] * 6,
        delta_tau_max=[args.delta_tau_max] * 6,
    )
    acados_export_dir = phase_acados_export_dir(args, export_suffix)
    solver = AcadosRTISolver(
        problem,
        config=AcadosRTIConfig(
            code_export_directory=acados_export_dir,
            qp_solver=args.acados_qp_solver,
            qp_solver_iter_max=args.acados_qp_solver_iter_max,
            nlp_solver_type=args.acados_nlp_solver_type,
            regularization=args.regularization,
            fast_control=args.fast_control,
            build_solver=not args.acados_runtime_only,
            verbose=args.acados_verbose,
        ),
        debug=args.debug,
    )
    return arm, problem, solver


def run_with_env(args, env, viewer=None):
    env.reset()
    robot = env.robot
    target, target_label = build_single_target(
        target_cylinder=args.target_cylinder,
        target_clearance=args.target_clearance,
        rotate_layout=not args.raw_pddl_targets,
    )
    approach_target = target.copy()
    initial_ee_pos = robot.ee_pos.copy()
    waypoint_path = build_parabolic_waypoints(
        initial_ee_pos,
        target,
        num_waypoints=args.num_waypoints,
    )
    waypoint_s = waypoint_arclengths(waypoint_path)
    use_velocity_refs = args.ee_velocity_weight > 0.0
    waypoint_index = 0
    closest_waypoint_index = 0
    distance_to_path = 0.0
    path_progress = 0.0
    arc_height = parabolic_arc_height(initial_ee_pos, target)

    arm, problem, solver = make_solver(
        args,
        env,
        initial_ee_pos,
        target,
    )

    dt = robot.model.opt.timestep
    mpc_every = max(int(round(args.mpc_dt / dt)), 1)
    draw_every = max(int(round(TRAJ_SAMPLE_DT / dt)), 1)

    current_tau = arm._clip_tau(arm.bias_for_state(arm.get_state()))
    current_tau = apply_optional_tau_limit(current_tau, args.apply_tau_limit)
    target_tau = current_tau.copy()
    problem.set_previous_tau(current_tau)
    last_mpc_success = True
    traj_actual = deque([initial_ee_pos.copy()], maxlen=TRAJ_MAX_PTS)
    target_marker_position = target.copy()
    max_ineq_violation = 0.0
    previous_ee_err = float(np.linalg.norm(robot.ee_pos - target))
    target_settle_active = False
    phase = "approach"
    approach_hold_count = 0
    bottom_hold_count = 0
    return_hold_count = 0
    real_time_wall_start = time.perf_counter()
    real_time_sim_start = float(robot.data.time)
    print(f"target: {target_label} {np.round(target, 4).tolist()}")
    if args.debug:
        print(
            f"waypoint path: total={len(waypoint_path)}, "
            f"path_length={waypoint_s[-1]:.4f}, "
            f"arc_height={arc_height:.4f}, "
            f"p0={np.round(waypoint_path[0], 4).tolist()}, "
            f"p_goal={np.round(waypoint_path[-1], 4).tolist()}"
        )

    for step in range(args.max_steps):
        if viewer is not None and not viewer.is_running():
            return 1

        ee_err = float(np.linalg.norm(robot.ee_pos - target))
        upright_err = terminal_upright_error(robot)
        active_reach_tol = (
            args.vertical_reach_tol
            if phase in {"descend", "ascend"}
            else args.reach_tol
        )
        if ee_err <= args.goal_hold_radius:
            target_settle_active = True
        if ee_err <= active_reach_tol and upright_err <= args.upright_reach_tol:
            if phase == "approach":
                if (not args.enable_cylinder_insertion
                        or approach_hold_count >= args.approach_hold_steps):
                    print(
                        f"reached {target_label} err={ee_err:.4f} m, "
                        f"upright_err={upright_err:.4f}"
                    )
                    if not args.enable_cylinder_insertion:
                        print(f"done in {step} sim steps")
                        return 0

                    target = approach_target.copy()
                    target[2] -= args.target_clearance + args.cylinder_entry_depth
                    target_label = (
                        f"inside {args.target_cylinder} "
                        f"{args.cylinder_entry_depth:.3f} m below top"
                    )
                    waypoint_path = build_linear_waypoints(
                        robot.ee_pos.copy(),
                        target,
                        num_waypoints=args.vertical_num_waypoints,
                    )
                    waypoint_s = waypoint_arclengths(waypoint_path)
                    waypoint_index = 0
                    closest_waypoint_index = 0
                    distance_to_path = 0.0
                    path_progress = 0.0
                    arc_height = 0.0
                    target_settle_active = False
                    previous_ee_err = float(np.linalg.norm(robot.ee_pos - target))
                    target_marker_position = target.copy()
                    arm, problem, solver = make_solver(
                        args,
                        env,
                        robot.ee_pos.copy(),
                        target,
                        export_suffix="vertical",
                    )
                    problem.set_previous_tau(current_tau)
                    phase = "descend"
                    print(
                        f"starting vertical insertion target "
                        f"{np.round(target, 4).tolist()}"
                    )
                    continue
                approach_hold_count += 1
                target_settle_active = True
            elif phase == "descend":
                if bottom_hold_count >= args.vertical_hold_steps:
                    print(
                        f"inserted {target_label} err={ee_err:.4f} m, "
                        f"upright_err={upright_err:.4f}"
                    )
                    target = approach_target.copy()
                    target_label = f"return above {args.target_cylinder}"
                    waypoint_path = build_linear_waypoints(
                        robot.ee_pos.copy(),
                        target,
                        num_waypoints=args.vertical_num_waypoints,
                    )
                    waypoint_s = waypoint_arclengths(waypoint_path)
                    waypoint_index = 0
                    closest_waypoint_index = 0
                    distance_to_path = 0.0
                    path_progress = 0.0
                    arc_height = 0.0
                    target_settle_active = False
                    previous_ee_err = float(np.linalg.norm(robot.ee_pos - target))
                    target_marker_position = target.copy()
                    problem.set_previous_tau(current_tau)
                    phase = "ascend"
                    print(
                        f"starting vertical return target "
                        f"{np.round(target, 4).tolist()}"
                    )
                    continue
                bottom_hold_count += 1
                target_settle_active = True
            elif phase == "ascend":
                if return_hold_count >= args.vertical_return_hold_steps:
                    print(
                        f"returned {target_label} err={ee_err:.4f} m, "
                        f"upright_err={upright_err:.4f}"
                    )
                    print(f"done in {step} sim steps")
                    return 0
                return_hold_count += 1
                target_settle_active = True

        if step % mpc_every == 0:
            active_ref_speed = (
                args.vertical_ref_speed
                if phase in {"descend", "ascend"}
                else args.ref_speed
            )
            problem.set_box_active_mask(
                np.ones(args.horizon + 1, dtype=bool))
            problem.set_box_contact_allowed_mask(
                np.zeros(args.horizon + 1, dtype=bool))
            settle_to_target = target_settle_active
            if settle_to_target:
                refs = np.repeat(target[None, :], args.horizon + 1, axis=0)
                ref_distances = np.full(args.horizon + 1, waypoint_s[-1])
            else:
                ref_distances = reference_horizon_distances(
                    path_progress,
                    args.horizon,
                    args.mpc_dt,
                    active_ref_speed,
                    lookahead=args.reference_lookahead,
                    total_distance=waypoint_s[-1],
                )
                refs = sample_waypoint_path(
                    waypoint_path,
                    waypoint_s,
                    ref_distances,
                )
            if not use_velocity_refs:
                problem.set_reference(refs)
            else:
                if settle_to_target:
                    v_refs = np.zeros_like(refs)
                else:
                    v_refs = reference_velocities_from_path(
                        waypoint_path,
                        waypoint_s,
                        ref_distances,
                        active_ref_speed,
                    )
                problem.set_reference(refs, v_refs)
            max_delta_tau = args.tau_slew_rate * args.mpc_dt
            problem.set_previous_tau(current_tau)
            mpc_tau, _, diag = solver.step(arm.get_state())
            mpc_usable = (
                not diag.fallback_used
                and diag.inequality_violation_after
                >= -args.freeze_ineq_violation
            )
            desired_tau = mpc_tau
            last_mpc_success = mpc_usable
            max_ineq_violation = min(
                max_ineq_violation,
                diag.inequality_violation_after,
            )
            desired_tau = apply_optional_tau_limit(
                arm._clip_tau(desired_tau),
                args.apply_tau_limit,
            )
            target_tau = limit_torque_slew(
                desired_tau,
                current_tau,
                max_delta_tau,
            )
            if args.debug:
                dist_to_wp = float(np.linalg.norm(
                    robot.ee_pos - waypoint_path[waypoint_index]))
                print(
                    f"waypoint_index={waypoint_index}/{len(waypoint_path) - 1}, "
                    f"closest_waypoint_index={closest_waypoint_index}, "
                    f"path_progress={path_progress:.4f}/{waypoint_s[-1]:.4f}, "
                    f"distance_to_path={distance_to_path:.4f}, "
                    f"distance_to_current_waypoint={dist_to_wp:.4f}, "
                    f"distance_to_target={ee_err:.4f}, "
                    f"upright_error={upright_err:.4f}, "
                    f"p_ref_0={np.round(refs[0], 4).tolist()}, "
                    f"p_ref_N={np.round(refs[-1], 4).tolist()}, "
                    f"arc_height={arc_height:.4f}"
                )
                print(
                    "controller=mpc_acados, "
                    f"desired tau={np.round(desired_tau, 3).tolist()}, "
                    f"target tau={np.round(target_tau, 3).tolist()}, "
                    f"apply tau={np.round(current_tau, 3).tolist()}"
                )

        current_tau = limit_torque_slew(
            target_tau,
            current_tau,
            args.tau_slew_rate * dt,
        )
        env.step(current_tau)
        post_step_err = float(np.linalg.norm(robot.ee_pos - target))
        if post_step_err <= args.goal_hold_radius:
            progress_speed = 0.0
        else:
            progress_speed = (
                args.vertical_ref_speed
                if phase in {"descend", "ascend"}
                else args.ref_speed
            )
        if post_step_err > previous_ee_err + args.progress_error_slack:
            progress_speed = 0.0
        previous_ee_err = post_step_err
        path_progress, waypoint_index, closest_waypoint_index, distance_to_path = \
            update_path_progress(
            robot.ee_pos,
            waypoint_path,
            waypoint_s,
            path_progress,
            dt,
            progress_speed if last_mpc_success else 0.0,
            args.max_path_lead,
            args.waypoint_tracking_tol,
        )

        if viewer is not None and step % draw_every == 0:
            next_geom = 0
            traj_actual.append(robot.ee_pos.copy())
            next_geom = draw_polyline(
                viewer,
                next_geom,
                traj_actual,
                rgba=(0.55, 0.55, 0.55, 1.0),
                width=args.trace_width,
            )
            next_geom = draw_sphere_marker(
                viewer,
                next_geom,
                target_marker_position,
                rgba=(0.0, 0.1, 1.0, 1.0),
                radius=args.target_marker_radius,
            )
            next_geom = draw_sphere_marker(
                viewer,
                next_geom,
                robot.ee_pos,
                rgba=(0.0, 1.0, 0.75, 1.0),
                radius=args.tcp_marker_radius,
            )
            viewer.user_scn.ngeom = next_geom
            viewer.sync()
        if viewer is not None and args.real_time:
            pace_realtime(
                robot,
                real_time_wall_start,
                real_time_sim_start,
                args.real_time_factor,
            )

    final_err = float(np.linalg.norm(robot.ee_pos - target))
    final_upright_err = terminal_upright_error(robot)
    print(
        f"not finished, target={target_label!r}, "
        f"final_err={final_err:.4f} m, "
        f"final_upright_err={final_upright_err:.4f}"
    )
    print(f"max inequality violation: {max_ineq_violation:.4f}")
    return 1


def run_viewer(args):
    import mujoco.viewer

    env = environment("ur10e")
    env.reset()
    with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as viewer:
        viewer.opt.sitegroup[:] = 0
        viewer.sync()
        status = run_with_env(args, env, viewer)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
        return status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Move the UR10e end effector to one visible scene target with RTI NMPC."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--target-cylinder",
                        choices=sorted(PDDL_CYLINDER_RADII),
                        default=DEFAULT_TARGET_CYLINDER)
    parser.add_argument("--target-clearance", type=float, default=0.10,
                        help="meters above the cylinder top center")
    parser.add_argument("--reach-tol", type=float, default=0.01)
    parser.add_argument("--upright-reach-tol", type=float, default=0.25,
                        help="maximum terminal TCP vertical-axis error")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--mpc-dt", type=float, default=0.04)
    parser.add_argument("--ref-speed", type=float, default=0.18,
                        help="meters per second along the fixed Cartesian arc")
    parser.add_argument("--enable-cylinder-insertion",
                        dest="enable_cylinder_insertion",
                        action="store_true",
                        default=True,
                        help="after reaching the first target, descend into the cylinder and return")
    parser.add_argument("--no-cylinder-insertion",
                        dest="enable_cylinder_insertion",
                        action="store_false",
                        help="stop after the first target as in the original PDDL playback")
    parser.add_argument("--cylinder-entry-depth", type=float, default=0.04,
                        help="meters below the target cylinder top for the vertical insertion")
    parser.add_argument("--vertical-ref-speed", type=float, default=0.08,
                        help="meters per second for the insertion and return MPC")
    parser.add_argument("--vertical-reach-tol", type=float, default=0.015,
                        help="position tolerance for insertion and return phases")
    parser.add_argument("--approach-hold-steps", type=int, default=120,
                        help="simulation steps to hold above the cylinder before descending")
    parser.add_argument("--vertical-hold-steps", type=int, default=180,
                        help="simulation steps to hold inside the cylinder before returning")
    parser.add_argument("--vertical-return-hold-steps", type=int, default=120,
                        help="simulation steps to hold after returning above the cylinder")
    parser.add_argument("--vertical-num-waypoints", type=int, default=24,
                        help="linear waypoint count for vertical insertion and return")
    parser.add_argument("--num-waypoints", type=int, default=80)
    parser.add_argument("--waypoint-tracking-tol", type=float, default=0.025)
    parser.add_argument("--max-path-lead", type=float, default=0.10,
                        help="maximum arc-length lead of p_ref_0 over the closest path point")
    parser.add_argument("--reference-lookahead", type=float, default=0.03,
                        help="arc-length lookahead added to p_ref_0")
    parser.add_argument("--goal-hold-radius", type=float, default=0.08,
                        help="switch to fixed target and zero velocity references inside this radius")
    parser.add_argument("--progress-error-slack", type=float, default=0.002,
                        help="freeze path progress when target error increases by more than this")
    parser.add_argument("--freeze-ineq-violation", type=float, default=0.005,
                        help="freeze path progress when MPC inequality violation is below -value")
    parser.add_argument("--ee-pos-weight", type=float, default=500.0)
    parser.add_argument("--ee-z-weight", type=float, default=700.0)
    parser.add_argument("--ee-terminal-weight", type=float, default=3000.0)
    parser.add_argument("--ee-terminal-z-weight", type=float, default=4200.0)
    parser.add_argument("--ee-upright-weight", type=float, default=800.0,
                        help="running cost for keeping the TCP axis perpendicular to the ground")
    parser.add_argument("--ee-terminal-upright-weight", type=float,
                        default=1200.0)
    parser.add_argument("--qv-weight", type=float, default=0.08)
    parser.add_argument("--q-weight", type=float, default=0.0)
    parser.add_argument("--qf-weight", type=float, default=0.0)
    parser.add_argument("--qvf-weight", type=float, default=0.10)
    parser.add_argument("--ee-velocity-weight", type=float, default=2.0)
    parser.add_argument("--delta-tau-cost", type=float, default=0.02)
    parser.add_argument("--d-ground", type=float, default=0.02,
                        help="ground clearance constraint in meters; negative values effectively disable it")
    parser.add_argument("--d-safe", type=float, default=0.04)
    parser.add_argument("--d-box", type=float, default=0.03)
    parser.add_argument("--include-box", dest="include_box", action="store_true",
                        default=False,
                        help="include the table/box obstacle in MPC collision constraints")
    parser.add_argument("--no-include-box", dest="include_box",
                        action="store_false",
                        help="disable the table/box obstacle in MPC collision constraints")
    parser.add_argument("--collision-constraints", action="store_true",
                        default=True,
                        help="enable collision/ground constraints inside the MPC")
    parser.add_argument("--no-collision-constraints",
                        dest="collision_constraints",
                        action="store_false",
                        help="disable collision/ground constraints for smoother playback")
    parser.add_argument("--avoid-target-cylinder", action="store_true",
                        help="also treat the selected target cylinder as an obstacle")
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument(
        "--acados-export-dir",
        default=default_acados_export_dir(),
                        help="deprecated alias for --acados-above-export-dir")
    parser.add_argument("--acados-above-export-dir",
                        default=default_acados_export_dir("ur10e_above"),
                        help="prebuilt acados solver directory for the above/approach phase")
    parser.add_argument("--acados-insert-export-dir",
                        default=default_acados_export_dir("ur10e_insert"),
                        help="prebuilt acados solver directory for the vertical insert/return phase")
    parser.add_argument("--acados-qp-solver", default="PARTIAL_CONDENSING_HPIPM",
                        help="acados QP solver")
    parser.add_argument("--acados-qp-solver-iter-max", type=int, default=200,
                        help="maximum acados QP iterations")
    parser.add_argument("--acados-nlp-solver-type", default="SQP_RTI",
                        choices=["SQP_RTI", "SQP"],
                        help="acados NLP solver type")
    parser.add_argument("--acados-verbose", action="store_true",
                        help="print acados generation and solver output")
    parser.add_argument("--fast-control", action="store_true",
                        help="skip expensive post-solve diagnostics in the realtime MPC loop")
    parser.add_argument("--acados-runtime-only", action="store_true",
                        help="load prebuilt acados solvers and fail instead of generating/building at runtime")
    parser.add_argument("--delta-q-max", type=float, default=0.12)
    parser.add_argument("--delta-dq-max", type=float, default=0.45)
    parser.add_argument("--delta-tau-max", type=float, default=25.0)
    parser.add_argument("--apply-tau-limit", type=float, default=0.0,
                        help="optional symmetric torque clamp; <=0 uses model actuator limits only")
    parser.add_argument("--tau-slew-rate", type=float, default=120.0,
                        help="maximum applied torque change in Nm/s; <=0 disables")
    parser.add_argument("--real-time", dest="real_time", action="store_true",
                        default=True,
                        help="pace viewer playback to simulation time")
    parser.add_argument("--no-real-time", dest="real_time",
                        action="store_false",
                        help="run the viewer loop as fast as possible")
    parser.add_argument("--real-time-factor", type=float, default=1.0,
                        help="viewer playback speed multiplier")
    parser.add_argument("--wrap-joints", action="store_true")
    parser.add_argument("--raw-pddl-targets", action="store_true",
                        help="use raw PDDL x/y target instead of the rotated scene frame")
    parser.add_argument("--trace-width", type=float, default=0.0025,
                        help="width of the drawn end-effector trajectory line")
    parser.add_argument("--target-marker-radius", type=float, default=0.007,
                        help="radius of the blue marker showing the EE target position")
    parser.add_argument("--tcp-marker-radius", type=float, default=0.004,
                        help="radius of the cyan marker showing robot.ee_pos")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.acados_export_dir != default_acados_export_dir():
        args.acados_above_export_dir = args.acados_export_dir
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.headless:
        raise SystemExit(run_with_env(args, environment("ur10e")))
    raise SystemExit(run_viewer(args))
