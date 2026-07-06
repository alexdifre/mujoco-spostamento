#!/usr/bin/env python3
"""
acados-backed RTI solver for the MuJoCo-based arm NMPC problem.

The repository's problem model is numerical: MuJoCo supplies dynamics,
forward kinematics, and Jacobians at the current warm-start trajectory.
This module keeps that modeling layer and uses acados to solve the local
linear-quadratic RTI subproblem.
"""
from dataclasses import dataclass, field
import json
import os
import sys
import time

import numpy as np

from rti_sqp_mpc import RTIDiagnostics, Trajectory


def _default_acados_source_dir():
    short_path = r"C:\Users\ALESSA~1\acados"
    configured = os.environ.get("ACADOS_SOURCE_DIR", short_path)
    if " " in configured and os.path.isdir(short_path):
        return short_path
    return configured


ACADOS_SOURCE_DIR = _default_acados_source_dir()
ACADOS_TEMPLATE_DIR = os.path.join(
    ACADOS_SOURCE_DIR,
    "interfaces",
    "acados_template",
)
ACADOS_BIN_DIR = os.path.join(ACADOS_SOURCE_DIR, "bin")
ACADOS_LIB_DIR = os.path.join(ACADOS_SOURCE_DIR, "lib")
MINGW_BIN_DIR = r"C:\conda-forge\envs\mlc-stack\Library\mingw-w64\bin"


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_code_export_directory():
    return os.path.join(_repo_root(), "acados_generated", "ur10e_rti")


def _windows_short_path(path):
    if os.name != "nt":
        return path
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return path

    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    GetShortPathNameW.restype = wintypes.DWORD

    length = GetShortPathNameW(path, None, 0)
    if length == 0:
        return path
    buffer = ctypes.create_unicode_buffer(length)
    result = GetShortPathNameW(path, buffer, length)
    if result == 0 or result > length:
        return path
    return buffer.value


def _windows_path_without_spaces(path):
    if os.name != "nt" or " " not in path:
        return path
    root = _repo_root()
    try:
        rel_path = os.path.relpath(path, root)
    except ValueError:
        return _windows_short_path(path)
    if rel_path.startswith(".."):
        return _windows_short_path(path)
    short_root = _windows_short_path(root)
    return os.path.join(short_root, rel_path)


def configure_acados_environment():
    if os.path.isdir(ACADOS_TEMPLATE_DIR) and ACADOS_TEMPLATE_DIR not in sys.path:
        sys.path.insert(0, ACADOS_TEMPLATE_DIR)
    os.environ["ACADOS_SOURCE_DIR"] = ACADOS_SOURCE_DIR
    if os.path.isdir(ACADOS_BIN_DIR):
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if ACADOS_BIN_DIR not in path_entries:
            os.environ["PATH"] = ACADOS_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            add_dll_directory(ACADOS_BIN_DIR)
    if os.path.isdir(ACADOS_LIB_DIR):
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if ACADOS_LIB_DIR not in path_entries:
            os.environ["PATH"] = ACADOS_LIB_DIR + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            add_dll_directory(ACADOS_LIB_DIR)
    if os.path.isdir(MINGW_BIN_DIR):
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if MINGW_BIN_DIR not in path_entries:
            os.environ["PATH"] = MINGW_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


@dataclass
class AcadosRTIConfig:
    code_export_directory: str = field(default_factory=_default_code_export_directory)
    qp_solver: str = "FULL_CONDENSING_DAQP"
    qp_solver_iter_max: int = 200
    nlp_solver_type: str = "SQP_RTI"
    hessian_approx: str = "EXACT"
    regularization: float = 1e-8
    nlp_solver_warm_start_first_qp: bool = True
    nlp_solver_warm_start_first_qp_from_nlp: bool = True
    fast_control: bool = False
    build_solver: bool = False
    verbose: bool = False


class AcadosRTISolver:
    def __init__(self, problem, config=None, debug=False):
        configure_acados_environment()
        self.problem = problem
        self.config = config or AcadosRTIConfig()
        export_dir = os.path.abspath(self.config.code_export_directory)
        os.makedirs(export_dir, exist_ok=True)
        self.config.code_export_directory = _windows_path_without_spaces(export_dir)
        self.debug = bool(debug)
        self.previous_trajectory = None
        self.mpc_step = 0
        self.nx = int(problem.nx)
        self.nu = int(problem.nu)
        self.nz = self.nx + self.nu
        self.nw = self.nz + self.nu
        self.N = int(problem.horizon)
        self.stage_rows = 3 + 3 + 3 + self.problem.n + self.problem.n + self.nu + self.nu
        self.terminal_rows = 3 + 3 + 3 + self.problem.n + self.problem.n
        self.ng = self._count_stage_inequalities()
        self._build_solver()

    def _count_stage_inequalities(self):
        count = 0
        if self.problem.collision_model is not None:
            count += int(self.problem.collision_model.residuals(
                self.problem.q_nominal,
                include_box=True,
                allow_box_contact_geometry=False,
            ).size)
        return count

    @property
    def _u_base(self):
        return (self.N + 1) * self.nx

    @property
    def _dyn_param_size(self):
        return self.nx * self.nx + self.nx * self.nu + self.nx

    @property
    def _stage_cost_param_size(self):
        return self.nw * self.nw + self.nw

    @property
    def _terminal_cost_param_size(self):
        return self.nz * self.nz + self.nz

    @property
    def _param_size(self):
        return (
            self._dyn_param_size
            + self._stage_cost_param_size
            + self._terminal_cost_param_size
        )

    def _build_solver(self):
        import casadi as ca
        from acados_template import (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ocp_get_default_cmake_builder,
        )

        json_file = os.path.join(self.config.code_export_directory, "acados_ocp.json")
        dll_file = os.path.join(
            self.config.code_export_directory,
            "acados_ocp_solver_ur10e_acados_rti.dll",
        )
        signature_file = os.path.join(
            self.config.code_export_directory,
            "solver_signature.json",
        )
        warm_start_from_nlp = self._supports_warm_start_from_nlp()
        signature = {
            "N": self.N,
            "nx": self.nx,
            "nu": self.nu,
            "nz": self.nz,
            "nw": self.nw,
            "ng": self.ng,
            "param_size": self._param_size,
            "qp_solver": self.config.qp_solver,
            "nlp_solver_type": self.config.nlp_solver_type,
            "nlp_solver_warm_start_first_qp":
                self.config.nlp_solver_warm_start_first_qp,
            "nlp_solver_warm_start_first_qp_from_nlp": warm_start_from_nlp,
            "constraint_formulation": "hard",
        }
        if os.path.exists(json_file) and os.path.exists(dll_file) and os.path.exists(signature_file):
            with open(signature_file, "r", encoding="utf-8") as f:
                old_signature = json.load(f)
            if old_signature == signature:
                self._refresh_existing_json_paths(json_file)
                self.solver = AcadosOcpSolver(
                    None,
                    json_file=json_file,
                    generate=False,
                    build=False,
                    verbose=self.config.verbose,
                )
                return

        if not self.config.build_solver:
            raise RuntimeError(
                "precompiled acados solver is missing or stale: "
                f"{self.config.code_export_directory}. "
                "Pass --build-acados explicitly to regenerate it."
            )

        z = ca.SX.sym("z", self.nz)
        du = ca.SX.sym("du", self.nu)
        p = ca.SX.sym("p", self._param_size)

        cursor = 0
        A = ca.reshape(p[cursor:cursor + self.nx * self.nx], self.nx, self.nx)
        cursor += self.nx * self.nx
        B = ca.reshape(p[cursor:cursor + self.nx * self.nu], self.nx, self.nu)
        cursor += self.nx * self.nu
        c = p[cursor:cursor + self.nx]
        cursor += self.nx

        H = ca.reshape(p[cursor:cursor + self.nw * self.nw], self.nw, self.nw)
        cursor += self.nw * self.nw
        g = p[cursor:cursor + self.nw]
        cursor += self.nw

        H_e = ca.reshape(p[cursor:cursor + self.nz * self.nz], self.nz, self.nz)
        cursor += self.nz * self.nz
        g_e = p[cursor:cursor + self.nz]

        dx = z[:self.nx]
        z_next = ca.vertcat(A @ dx + B @ du + c, du)
        w = ca.vertcat(z, du)

        model = AcadosModel()
        model.name = "ur10e_acados_rti"
        model.x = z
        model.u = du
        model.p = p
        model.disc_dyn_expr = z_next
        model.cost_expr_ext_cost = 0.5 * ca.mtimes([w.T, H, w]) + g.T @ w
        model.cost_expr_ext_cost_e = 0.5 * ca.mtimes([z.T, H_e, z]) + g_e.T @ z

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.parameter_values = np.zeros(self._param_size)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.constraints.x0 = np.zeros(self.nz)
        ocp.constraints.idxbx = np.arange(self.nz)
        ocp.constraints.lbx = -1e9 * np.ones(self.nz)
        ocp.constraints.ubx = 1e9 * np.ones(self.nz)
        ocp.constraints.idxbx_e = np.arange(self.nz)
        ocp.constraints.lbx_e = -1e9 * np.ones(self.nz)
        ocp.constraints.ubx_e = 1e9 * np.ones(self.nz)
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = -1e9 * np.ones(self.nu)
        ocp.constraints.ubu = 1e9 * np.ones(self.nu)
        if self.ng:
            ocp.constraints.C = np.zeros((self.ng, self.nz))
            ocp.constraints.D = np.zeros((self.ng, self.nu))
            ocp.constraints.lg = -1e9 * np.ones(self.ng)
            ocp.constraints.ug = 1e9 * np.ones(self.ng)
            ocp.constraints.C_e = np.zeros((self.ng, self.nz))
            ocp.constraints.lg_e = -1e9 * np.ones(self.ng)
            ocp.constraints.ug_e = 1e9 * np.ones(self.ng)

        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.qp_solver = self.config.qp_solver
        ocp.solver_options.qp_solver_iter_max = self.config.qp_solver_iter_max
        ocp.solver_options.nlp_solver_type = self.config.nlp_solver_type
        ocp.solver_options.hessian_approx = self.config.hessian_approx
        self._set_optional_solver_option(
            ocp.solver_options,
            "nlp_solver_warm_start_first_qp",
            bool(self.config.nlp_solver_warm_start_first_qp),
        )
        self._set_optional_solver_option(
            ocp.solver_options,
            "nlp_solver_warm_start_first_qp_from_nlp",
            bool(warm_start_from_nlp),
        )
        ocp.solver_options.tf = float(self.N)
        ocp.solver_options.print_level = 1 if self.config.verbose else 0
        ocp.code_export_directory = self.config.code_export_directory

        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.additional_cmake_options = ""
        cmake_builder.build_dir = os.path.join(
            self.config.code_export_directory,
            "build_active",
        )

        self.solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            cmake_builder=cmake_builder,
            verbose=self.config.verbose,
        )
        os.makedirs(self.config.code_export_directory, exist_ok=True)
        with open(signature_file, "w", encoding="utf-8") as f:
            json.dump(signature, f, indent=2)

    @staticmethod
    def _set_optional_solver_option(options, name, value):
        try:
            setattr(options, name, value)
        except Exception:
            pass

    def _supports_warm_start_from_nlp(self):
        return (
            bool(self.config.nlp_solver_warm_start_first_qp_from_nlp)
            and self.config.qp_solver == "PARTIAL_CONDENSING_HPIPM"
        )

    def _refresh_existing_json_paths(self, json_file):
        """Keep generated acados JSON loadable after moving the repository."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        changed = False
        code_gen_opts = data.get("code_gen_opts", {})
        if code_gen_opts.get("code_export_directory") != self.config.code_export_directory:
            code_gen_opts["code_export_directory"] = self.config.code_export_directory
            data["code_gen_opts"] = code_gen_opts
            changed = True
        if code_gen_opts.get("json_file") != json_file:
            code_gen_opts["json_file"] = json_file
            data["code_gen_opts"] = code_gen_opts
            changed = True
        if changed:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

    def _warm_start(self, measured_state):
        if self.previous_trajectory is None:
            return self.problem.make_initial_trajectory(measured_state)
        return self.problem.shift_trajectory(self.previous_trajectory, measured_state)

    def _stage_cost_quadratic(self, residuals, jacobian, k):
        row0 = k * self.stage_rows
        row1 = row0 + self.stage_rows
        rows = slice(row0, row1)
        r = residuals[rows]
        J = np.zeros((self.stage_rows, self.nw), dtype=np.float64)

        x_cols = slice(k * self.nx, (k + 1) * self.nx)
        u_cols = slice(self._u_base + k * self.nu, self._u_base + (k + 1) * self.nu)
        J[:, :self.nx] = jacobian[rows, x_cols].toarray()
        J[:, self.nz:] = jacobian[rows, u_cols].toarray()
        if k > 0:
            prev_cols = slice(
                self._u_base + (k - 1) * self.nu,
                self._u_base + k * self.nu,
            )
            J[:, self.nx:self.nz] = jacobian[rows, prev_cols].toarray()

        H = J.T @ J + self.config.regularization * np.eye(self.nw)
        g = J.T @ r
        return H, g

    def _terminal_cost_quadratic(self, residuals, jacobian):
        row0 = self.N * self.stage_rows
        row1 = row0 + self.terminal_rows
        rows = slice(row0, row1)
        r = residuals[rows]
        J = np.zeros((self.terminal_rows, self.nz), dtype=np.float64)
        x_cols = slice(self.N * self.nx, (self.N + 1) * self.nx)
        J[:, :self.nx] = jacobian[rows, x_cols].toarray()
        H = J.T @ J + self.config.regularization * np.eye(self.nz)
        g = J.T @ r
        return H, g

    def _pack_params(self, A=None, B=None, c=None, H=None, g=None,
                     H_e=None, g_e=None):
        values = []
        values.append(np.zeros((self.nx, self.nx)) if A is None else A)
        values.append(np.zeros((self.nx, self.nu)) if B is None else B)
        values.append(np.zeros(self.nx) if c is None else c)
        values.append(np.zeros((self.nw, self.nw)) if H is None else H)
        values.append(np.zeros(self.nw) if g is None else g)
        values.append(np.zeros((self.nz, self.nz)) if H_e is None else H_e)
        values.append(np.zeros(self.nz) if g_e is None else g_e)
        packed = []
        for value in values:
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 2:
                packed.append(array.reshape(-1, order="F"))
            else:
                packed.append(array.reshape(-1))
        return np.concatenate(packed)

    def _set_bounds(self, delta_lower, delta_upper):
        for k in range(self.N + 1):
            if k == 0:
                lbx = np.zeros(self.nz)
                ubx = np.zeros(self.nz)
            else:
                lbx = -1e9 * np.ones(self.nz)
                ubx = 1e9 * np.ones(self.nz)
                x_slice = slice(k * self.nx, (k + 1) * self.nx)
                lbx[:self.nx] = delta_lower[x_slice]
                ubx[:self.nx] = delta_upper[x_slice]
                prev_slice = slice(
                    self._u_base + (k - 1) * self.nu,
                    self._u_base + k * self.nu,
                )
                lbx[self.nx:self.nz] = delta_lower[prev_slice]
                ubx[self.nx:self.nz] = delta_upper[prev_slice]
            if k < self.N:
                self.solver.constraints_set(k, "lbx", lbx)
                self.solver.constraints_set(k, "ubx", ubx)
            else:
                self.solver.constraints_set(k, "lbx", lbx)
                self.solver.constraints_set(k, "ubx", ubx)

        for k in range(self.N):
            u_slice = slice(self._u_base + k * self.nu, self._u_base + (k + 1) * self.nu)
            self.solver.constraints_set(k, "lbu", delta_lower[u_slice])
            self.solver.constraints_set(k, "ubu", delta_upper[u_slice])

    def _set_linearized_inequalities(self, values, jacobian):
        if not self.ng:
            return
        if values.size != (self.N + 1) * self.ng:
            raise ValueError(
                "acados inequality layout mismatch: expected "
                f"{(self.N + 1) * self.ng}, got {values.size}"
            )
        for k in range(self.N + 1):
            rows = slice(k * self.ng, (k + 1) * self.ng)
            x_cols = slice(k * self.nx, (k + 1) * self.nx)
            C = np.zeros((self.ng, self.nz), dtype=np.float64)
            C[:, :self.nx] = jacobian[rows, x_cols].toarray()
            self.solver.constraints_set(k, "C", C, api="new")
            self.solver.constraints_set(k, "lg", -values[rows])
            self.solver.constraints_set(k, "ug", 1e9 * np.ones(self.ng))

    def _initialize_iterate(self, reset=False):
        if not reset:
            return
        for k in range(self.N + 1):
            self.solver.set(k, "x", np.zeros(self.nz))
        for k in range(self.N):
            self.solver.set(k, "u", np.zeros(self.nu))

    def step(self, measured_state):
        profile = {}
        total_start = time.perf_counter()

        def elapsed_ms(start):
            return 1000.0 * (time.perf_counter() - start)

        t0 = time.perf_counter()
        self.problem.prepare_step(measured_state)
        profile["prepare_step_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        had_previous = self.previous_trajectory is not None
        warm = self._warm_start(measured_state)
        warm_ms = elapsed_ms(t0)
        profile["rollout_or_warmstart_ms"] = warm_ms
        profile["trajectory_shift_ms"] = warm_ms if had_previous else 0.0

        t0 = time.perf_counter()
        y_bar = warm.stack()
        profile["trajectory_stack_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        kinematics_cache = self.problem.kinematics_cache_for_trajectory(warm)
        profile["kinematics_cache_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        dynamics_cache = self.problem.dynamics_cache_for_trajectory(warm)
        profile["dynamics_linearization_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        residuals, jacobian = self.problem.residuals_with_jacobian(
            y_bar,
            kinematics_cache=kinematics_cache,
        )
        profile["cost_residual_jacobian_ms"] = elapsed_ms(t0)

        profile["inequality_value_ms"] = 0.0
        t0 = time.perf_counter()
        ineq_before, jac_ineq = self.problem.inequality_constraints_with_jacobian(
            y_bar,
            kinematics_cache=kinematics_cache,
        )
        profile["inequality_jacobian_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        delta_lower, delta_upper = self.problem.delta_bounds(y_bar)
        profile["bounds_delta_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        terminal_H, terminal_g = self._terminal_cost_quadratic(residuals, jacobian)
        profile["terminal_cost_ms"] = elapsed_ms(t0)

        cost_set_ms = 0.0
        dynamics_set_ms = 0.0
        for k in range(self.N):
            A = dynamics_cache.A[k]
            B = dynamics_cache.B[k]
            c = dynamics_cache.x_next[k] - warm.x[k + 1]

            t_cost = time.perf_counter()
            H, g = self._stage_cost_quadratic(residuals, jacobian, k)
            cost_set_ms += elapsed_ms(t_cost)

            t_dyn = time.perf_counter()
            self.solver.set(
                k,
                "p",
                self._pack_params(
                    A=A,
                    B=B,
                    c=c,
                    H=H,
                    g=g,
                    H_e=terminal_H,
                    g_e=terminal_g,
                ),
            )
            dynamics_set_ms += elapsed_ms(t_dyn)

        t_dyn = time.perf_counter()
        self.solver.set(
            self.N,
            "p",
            self._pack_params(H_e=terminal_H, g_e=terminal_g),
        )
        dynamics_set_ms += elapsed_ms(t_dyn)
        profile["acados_set_cost_ms"] = cost_set_ms
        profile["acados_set_dynamics_ms"] = dynamics_set_ms

        t0 = time.perf_counter()
        self._set_bounds(delta_lower, delta_upper)
        profile["acados_set_bounds_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        self._set_linearized_inequalities(ineq_before, jac_ineq)
        profile["acados_set_constraints_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        self._initialize_iterate(reset=self.previous_trajectory is None)
        profile["acados_initialize_iterate_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        status = int(self.solver.solve())
        profile["acados_solve_ms"] = elapsed_ms(t0)
        success = status == 0

        t0 = time.perf_counter()
        dx = np.vstack([self.solver.get(k, "x")[:self.nx] for k in range(self.N + 1)])
        du = np.vstack([self.solver.get(k, "u") for k in range(self.N)])
        x_new = warm.x + dx
        u_new = warm.u + du
        traj_new = Trajectory(x_new, u_new)
        applied = self.problem.arm._clip_tau(traj_new.u[0])
        profile["acados_get_solution_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        cost_before = 0.5 * float(residuals @ residuals)
        eq_before_norm = float("nan")
        ineq_before_violation = float(
            np.min(np.minimum(ineq_before, 0.0)) if ineq_before.size else 0.0)
        if self.config.fast_control:
            cost_after = float("nan")
            eq_after_norm = float("nan")
            ineq_after_violation = ineq_before_violation
        else:
            y_new = traj_new.stack()
            cost_after = float(self.problem.cost(y_new))
            eq_before = self.problem.equality_constraints(y_bar)
            eq_after = self.problem.equality_constraints(y_new)
            ineq_after = self.problem.inequality_constraints(y_new)
            eq_before_norm = float(np.linalg.norm(eq_before))
            eq_after_norm = float(np.linalg.norm(eq_after))
            ineq_after_violation = float(
                np.min(np.minimum(ineq_after, 0.0)) if ineq_after.size else 0.0)
        profile["post_solve_diagnostics_ms"] = elapsed_ms(t0)

        t0 = time.perf_counter()
        if success:
            self.previous_trajectory = traj_new
        else:
            self.previous_trajectory = None
            applied = self.problem.previous_tau.copy()
        profile["trajectory_commit_ms"] = elapsed_ms(t0)
        profile["total_profiled_ms"] = elapsed_ms(total_start)

        diag = RTIDiagnostics(
            mpc_step=self.mpc_step,
            cost_before=cost_before,
            equality_residual_norm_before=eq_before_norm,
            inequality_violation_before=ineq_before_violation,
            qp_status=f"acados_status_{status}",
            qp_iterations=0,
            qp_objective=0.0,
            delta_norm=float(np.linalg.norm(du) + np.linalg.norm(dx)),
            alpha=1.0,
            cost_after=cost_after,
            equality_residual_norm_after=eq_after_norm,
            inequality_violation_after=ineq_after_violation,
            applied_control=applied.copy(),
            sqp_steps=1,
            qp_solve_attempts=1,
            fallback_used=not success,
            profile_timings_ms=profile,
        )
        self.mpc_step += 1

        if self.debug:
            print(
                f"acados RTI step {diag.mpc_step}: "
                f"cost {diag.cost_before:.3e} -> {diag.cost_after:.3e}, "
                f"|C|={diag.equality_residual_norm_before:.3e}, "
                f"ineq={diag.inequality_violation_before:.3e}, "
                f"status={status}, |d|={diag.delta_norm:.3e}"
            )

        self.problem.set_previous_tau(applied)
        return applied, traj_new, diag
