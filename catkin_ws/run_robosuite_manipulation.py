import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / ".robosuite_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


@dataclass
class StepMetric:
    step: int
    stage: str
    eef_x: float
    eef_y: float
    eef_z: float
    cube_x: float
    cube_y: float
    cube_z: float
    target_x: float
    target_y: float
    target_z: float
    eef_error: float
    cube_lift: float
    reward: float
    action_norm: float
    success: bool
    grasped_internal: bool
    gripper_qpos_mean: float


def make_env(args):
    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robot)
    return suite.make(
        env_name="Lift",
        robots=args.robot,
        gripper_types=args.gripper,
        controller_configs=controller_config,
        has_renderer=args.viewer,
        has_offscreen_renderer=args.record_video,
        use_camera_obs=args.record_video,
        camera_names=args.camera,
        camera_heights=args.video_height,
        camera_widths=args.video_width,
        reward_shaping=True,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        ignore_done=True,
        initialization_noise=None,
        seed=args.seed,
    )


def position_action(eef_pos, target, gripper, action_dim, action_split, gain=7.0):
    delta = np.asarray(target, dtype=float) - np.asarray(eef_pos, dtype=float)
    action = np.zeros(action_dim, dtype=float)
    action[:3] = np.clip(gain * delta, -1.0, 1.0)
    for name, indexes in action_split.items():
        if "gripper" in name:
            action[indexes[0] : indexes[1]] = float(np.clip(gripper, -1.0, 1.0))
    return action


def infer_gripper_sign(env, steps=35):
    signs = {}
    for sign in (-1.0, 1.0):
        obs = env.reset()
        q0 = np.asarray(obs["robot0_gripper_qpos"], dtype=float).mean()
        for _ in range(steps):
            action = np.zeros(env.action_dim)
            action[-1] = sign
            obs, _, _, _ = env.step(action)
        q1 = np.asarray(obs["robot0_gripper_qpos"], dtype=float).mean()
        signs[sign] = q1 - q0
    close_sign = 1.0 if signs[1.0] >= signs[-1.0] else -1.0
    open_sign = -close_sign
    return open_sign, close_sign, signs


def run_episode(args):
    env = make_env(args)
    action_split = dict(env.robots[0]._action_split_indexes)
    open_sign, close_sign, sign_probe = infer_gripper_sign(env) if args.auto_gripper_sign else (-1.0, 1.0, {})
    if args.close_sign is not None:
        close_sign = float(np.sign(args.close_sign) or 1.0)
        open_sign = -close_sign
    obs = env.reset()

    initial_cube_pos = np.asarray(obs["cube_pos"], dtype=float)
    grasp_cube_pos = initial_cube_pos.copy()
    max_cube_z = float(initial_cube_pos[2])
    min_eef_error = math.inf
    reached_cube = False
    closed_gripper = False
    lifted = False
    metrics = []
    stage_counts = {}
    final_info = {}
    stage = "open"
    stable_grasp_steps = 0
    grasped_stable_steps = 0
    frames = []

    for step in range(args.max_steps):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        cube = np.asarray(obs["cube_pos"], dtype=float)
        max_cube_z = max(max_cube_z, float(cube[2]))

        above = grasp_cube_pos + np.array([0.0, 0.0, args.approach_height])
        grasp = grasp_cube_pos + np.array([0.0, 0.0, args.grasp_height])
        lift = grasp_cube_pos + np.array([0.0, 0.0, args.lift_height])
        grasp_error = float(np.linalg.norm(eef - grasp))

        if stage == "open" and step >= args.open_steps:
            stage = "approach"
        if stage == "approach" and np.linalg.norm(eef - above) <= args.approach_tol:
            stage = "descend"
        if stage == "descend":
            if grasp_error <= args.grasp_tol:
                stable_grasp_steps += 1
            else:
                stable_grasp_steps = 0
            if stable_grasp_steps >= args.preclose_hold_steps:
                stage = "close"
        if stage == "open":
            target = above
            gripper = open_sign
        elif stage == "approach":
            target = above
            gripper = open_sign
        elif stage == "descend":
            target = grasp
            gripper = open_sign
        elif stage == "close":
            target = grasp
            gripper = close_sign
            closed_gripper = True
        else:
            stage = "lift"
            target = lift
            gripper = close_sign

        action = position_action(
            eef,
            target,
            gripper,
            action_dim=env.action_dim,
            action_split=action_split,
            gain=args.position_gain,
        )
        obs, reward, done, info = env.step(action)

        eef_after = np.asarray(obs["robot0_eef_pos"], dtype=float)
        cube_after = np.asarray(obs["cube_pos"], dtype=float)
        err = float(np.linalg.norm(eef_after - target))
        lift_amount = float(cube_after[2] - initial_cube_pos[2])
        min_eef_error = min(min_eef_error, err)
        reached_cube = reached_cube or float(np.linalg.norm(eef_after - grasp)) <= args.grasp_tol
        lifted = lifted or lift_amount >= args.success_lift
        grasped_internal = bool(env._check_grasp(env.robots[0].gripper, env.cube)) if hasattr(env, "_check_grasp") else False
        success = bool((env._check_success() if hasattr(env, "_check_success") else False) or lifted)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        metrics.append(
            StepMetric(
                step=step,
                stage=stage,
                eef_x=float(eef_after[0]),
                eef_y=float(eef_after[1]),
                eef_z=float(eef_after[2]),
                cube_x=float(cube_after[0]),
                cube_y=float(cube_after[1]),
                cube_z=float(cube_after[2]),
                target_x=float(target[0]),
                target_y=float(target[1]),
                target_z=float(target[2]),
                eef_error=err,
                cube_lift=lift_amount,
                reward=float(reward),
                action_norm=float(np.linalg.norm(action)),
                success=success,
                grasped_internal=grasped_internal,
                gripper_qpos_mean=float(np.asarray(obs["robot0_gripper_qpos"], dtype=float).mean()),
            )
        )

        if args.viewer:
            env.render()
        if args.record_video:
            frame = obs.get(f"{args.camera}_image")
            if frame is not None:
                frames.append(np.flipud(frame).copy())

        if stage == "close":
            if grasped_internal:
                grasped_stable_steps += 1
            else:
                grasped_stable_steps = 0
            if grasped_stable_steps >= args.grasp_hold_steps or stage_counts.get("close", 0) >= args.close_steps:
                stage = "lift"

        if success and closed_gripper and lifted:
            final_info = dict(info)
            break
        final_info = dict(info)

    env.close()

    final = metrics[-1]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "environment": "robosuite Lift",
        "robot": args.robot,
        "gripper": args.gripper,
        "controller": args.controller,
        "seed": args.seed,
        "action_dim": env.action_dim,
        "action_split": action_split,
        "steps": len(metrics),
        "stage_counts": stage_counts,
        "initial_cube_pos": initial_cube_pos.tolist(),
        "final_cube_pos": [final.cube_x, final.cube_y, final.cube_z],
        "final_eef_pos": [final.eef_x, final.eef_y, final.eef_z],
        "min_eef_error_m": min_eef_error,
        "max_cube_lift_m": max(metric.cube_lift for metric in metrics),
        "final_cube_lift_m": final.cube_lift,
        "reached_cube": reached_cube,
        "closed_gripper": closed_gripper,
        "lifted_10cm": lifted,
        "grasped_internal_any": any(metric.grasped_internal for metric in metrics),
        "grasped_internal_final": final.grasped_internal,
        "success": bool(reached_cube and closed_gripper and lifted),
        "gripper_sign_probe": sign_probe,
        "final_info": final_info,
    }
    return report, metrics, frames


def write_outputs(report, metrics, frames, out_dir, fps):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"robosuite_grasp_metrics_{stamp}.csv"
    json_path = out_dir / f"robosuite_grasp_report_{stamp}.json"
    md_path = out_dir / f"robosuite_grasp_report_{stamp}.md"
    video_path = out_dir / f"robosuite_grasp_video_{stamp}.mp4" if frames else None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(StepMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if video_path is not None:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    lines = [
        "# Robosuite manipulation report",
        "",
        f"- success: {report['success']}",
        f"- robot: {report['robot']}",
        f"- gripper: {report['gripper']}",
        f"- controller: {report['controller']}",
        f"- seed: {report['seed']}",
        f"- steps: {report['steps']}",
        f"- reached cube: {report['reached_cube']}",
        f"- closed gripper: {report['closed_gripper']}",
        f"- lifted 10 cm: {report['lifted_10cm']}",
        f"- min eef error m: {report['min_eef_error_m']:.4f}",
        f"- max cube lift m: {report['max_cube_lift_m']:.4f}",
        f"- final cube lift m: {report['final_cube_lift_m']:.4f}",
        f"- initial cube pos: {report['initial_cube_pos']}",
        f"- final cube pos: {report['final_cube_pos']}",
        f"- metrics csv: {csv_path.name}",
        f"- raw json: {json_path.name}",
    ]
    if video_path is not None:
        lines.append(f"- video: {video_path.name}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Robosuite UR5e + Robotiq85 cube manipulation experiment.")
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--gripper", default="Robotiq85Gripper")
    parser.add_argument("--controller", default="BASIC")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open-steps", type=int, default=40)
    parser.add_argument("--descend-after", type=int, default=260)
    parser.add_argument("--close-after", type=int, default=430)
    parser.add_argument("--close-steps", type=int, default=100)
    parser.add_argument("--preclose-hold-steps", type=int, default=25)
    parser.add_argument("--grasp-hold-steps", type=int, default=4)
    parser.add_argument("--close-sign", type=float, default=None)
    parser.add_argument("--approach-height", type=float, default=0.12)
    parser.add_argument("--grasp-height", type=float, default=-0.015)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--success-lift", type=float, default=0.10)
    parser.add_argument("--approach-tol", type=float, default=0.035)
    parser.add_argument("--grasp-tol", type=float, default=0.045)
    parser.add_argument("--position-gain", type=float, default=4.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--auto-gripper-sign", action="store_true", default=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "robosuite_manipulation")
    return parser.parse_args()


def main():
    args = parse_args()
    report, metrics, frames = run_episode(args)
    csv_path, json_path, md_path, video_path = write_outputs(report, metrics, frames, args.out_dir, args.video_fps)
    print(json.dumps(report, indent=2))
    print(f"metrics_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    if video_path is not None:
        print(f"video={video_path}")
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
