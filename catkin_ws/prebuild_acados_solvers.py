#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from environment import environment
from run_pddl_plan import (
    build_single_target,
    default_acados_export_dir,
    make_solver as make_pddl_solver,
    parse_args as parse_pddl_args,
)


def _pddl_args(extra_args):
    return parse_pddl_args([
        "--headless",
        "--max-steps",
        "1",
        *extra_args,
    ])


def build_pddl_solver(extra_args, export_suffix=""):
    args = _pddl_args(extra_args)
    env = environment("ur10e")
    env.reset()
    target, _ = build_single_target(
        args.target_cylinder,
        args.target_clearance,
        rotate_layout=not args.raw_pddl_targets,
    )
    make_pddl_solver(
        args,
        env,
        env.robot.ee_pos.copy(),
        target,
        export_suffix=export_suffix,
    )


def build_default_pddl_set(args):
    common = [
        "--horizon",
        str(args.horizon),
        "--mpc-dt",
        str(args.mpc_dt),
        "--acados-above-export-dir",
        args.acados_above_export_dir,
        "--acados-insert-export-dir",
        args.acados_insert_export_dir,
        "--acados-qp-solver",
        args.acados_qp_solver,
        "--acados-qp-solver-iter-max",
        str(args.acados_qp_solver_iter_max),
        "--acados-nlp-solver-type",
        args.acados_nlp_solver_type,
        "--regularization",
        str(args.regularization),
        "--build-acados",
    ]

    print(f"building solver A: above/free-space -> {args.acados_above_export_dir}_free")
    build_pddl_solver([*common, "--no-collision-constraints"])

    print(f"building solver B: above/collision -> {args.acados_above_export_dir}")
    build_pddl_solver([*common, "--collision-constraints"])

    if args.include_vertical:
        print(f"building solver C: insert/collision -> {args.acados_insert_export_dir}")
        build_pddl_solver([*common, "--collision-constraints"], "vertical")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and compile acados solvers offline for runtime-only MPC."
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--mpc-dt", type=float, default=0.04)
    parser.add_argument("--acados-above-export-dir",
                        default=default_acados_export_dir("ur10e_above"))
    parser.add_argument("--acados-insert-export-dir",
                        default=default_acados_export_dir("ur10e_insert"))
    parser.add_argument("--acados-qp-solver", default="FULL_CONDENSING_DAQP")
    parser.add_argument("--acados-qp-solver-iter-max", type=int, default=200)
    parser.add_argument("--acados-nlp-solver-type", default="SQP_RTI",
                        choices=["SQP_RTI", "SQP"])
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--include-vertical", action="store_true",
                        help="also build the vertical/collision variant used during insertion")
    return parser.parse_args()


def main():
    args = parse_args()
    build_default_pddl_set(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
