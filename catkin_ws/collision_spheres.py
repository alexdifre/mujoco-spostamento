#!/usr/bin/env python3
"""
Collision-sphere environment constraints for NMPC.

Each robot link is approximated by spheres fixed in the corresponding MuJoCo
body frame. Constraints are numerical signed-distance residuals:

    positive  -> satisfied
    zero      -> active
    negative  -> violation
"""
from dataclasses import dataclass

import numpy as np
import mujoco


def _sdf_value_and_gradient(sdf, p):
    value_and_gradient = getattr(sdf, "value_and_gradient", None)
    if value_and_gradient is None:
        raise TypeError(
            f"{type(sdf).__name__} must implement value_and_gradient(p)"
        )
    dist, grad = value_and_gradient(p)
    return float(dist), np.asarray(grad, dtype=np.float64)


@dataclass(frozen=True)
class CollisionSphere:
    body_name: str
    local_pos: np.ndarray
    radius: float
    name: str = ""
    box_contact_geometry: bool = False

    def __post_init__(self):
        object.__setattr__(self, "local_pos",
                           np.asarray(self.local_pos, dtype=np.float64))


class FiniteCylinderSDF:
    """Signed distance to the lateral surface of a finite open cylinder.

    The MPC should avoid the cylinder wall, not the top/bottom disk areas.
    Points above or below the cylinder height are measured to the closest
    lateral rim instead of to a cap plane.
    """

    def __init__(self, center, radius, half_height):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.half_height = float(half_height)

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64)
        radial = np.linalg.norm(p[:2] - self.center[:2]) - self.radius
        vertical_excess = abs(float(p[2] - self.center[2])) - self.half_height
        if vertical_excess <= 0.0:
            return float(radial)
        return float(np.hypot(radial, vertical_excess))

    def value_and_gradient(self, p):
        p = np.asarray(p, dtype=np.float64)
        xy = p[:2] - self.center[:2]
        rho = float(np.linalg.norm(xy))
        radial = rho - self.radius
        z_delta = float(p[2] - self.center[2])
        vertical_excess = abs(z_delta) - self.half_height

        if rho > 1e-12:
            grad_radial = np.array([xy[0] / rho, xy[1] / rho, 0.0])
        else:
            grad_radial = np.array([1.0, 0.0, 0.0])

        if vertical_excess <= 0.0:
            return float(radial), grad_radial

        z_sign = 1.0 if z_delta >= 0.0 else -1.0
        grad_vertical = np.array([0.0, 0.0, z_sign])
        norm = float(np.hypot(radial, vertical_excess))
        if norm <= 1e-12:
            return 0.0, grad_radial
        gradient = (
            radial * grad_radial +
            vertical_excess * grad_vertical
        ) / norm
        return norm, gradient


class AxisAlignedBoxSDF:
    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype=np.float64)
        self.half_extents = np.asarray(half_extents, dtype=np.float64)

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64)
        q = np.abs(p - self.center) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return float(outside + inside)

    def value_and_gradient(self, p):
        p = np.asarray(p, dtype=np.float64)
        delta = p - self.center
        q = np.abs(delta) - self.half_extents
        signs = np.where(delta >= 0.0, 1.0, -1.0)
        positive = np.maximum(q, 0.0)
        outside_norm = float(np.linalg.norm(positive))
        if outside_norm > 1e-12:
            value = outside_norm
            gradient = positive * signs / outside_norm
            return value, gradient

        axis = int(np.argmax(q))
        gradient = np.zeros(3, dtype=np.float64)
        gradient[axis] = signs[axis]
        return float(np.max(q)), gradient


class CollisionSphereModel:
    def __init__(self, model, data, qpos_addr, spheres,
                 static_obstacles=None, box_sdf=None,
                 d_ground=0.02, d_safe=0.04, d_box=0.03,
                 dof_addr=None):
        self.model = model
        self.data = data
        self.qpos_addr = np.asarray(qpos_addr, dtype=np.int32)
        self.dof_addr = np.asarray(
            qpos_addr if dof_addr is None else dof_addr,
            dtype=np.int32,
        )
        self.spheres = list(spheres)
        self.static_obstacles = list(static_obstacles or [])
        self.box_sdf = box_sdf
        self.d_ground = float(d_ground)
        self.d_safe = float(d_safe)
        self.d_box = float(d_box)

        self.body_ids = []
        for sphere in self.spheres:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, sphere.body_name)
            if body_id < 0:
                raise ValueError(f"unknown body {sphere.body_name!r}")
            self.body_ids.append(body_id)

    def _with_q(self, q, callback):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self.qpos_addr] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return callback()
        finally:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            mujoco.mj_forward(self.model, self.data)

    def sphere_centers(self, q):
        q = np.asarray(q, dtype=np.float64)

        def compute():
            centers = []
            for sphere, body_id in zip(self.spheres, self.body_ids):
                R = self.data.xmat[body_id].reshape(3, 3)
                centers.append(self.data.xpos[body_id] + R @ sphere.local_pos)
            return np.asarray(centers, dtype=np.float64)

        return self._with_q(q, compute)

    def sphere_centers_and_jacobians(self, q):
        q = np.asarray(q, dtype=np.float64)

        def compute():
            centers = []
            jacobians = []
            for sphere, body_id in zip(self.spheres, self.body_ids):
                R = self.data.xmat[body_id].reshape(3, 3)
                center = self.data.xpos[body_id] + R @ sphere.local_pos
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jac(self.model, self.data, jacp, jacr,
                              center, body_id)
                centers.append(center.copy())
                jacobians.append(jacp[:, self.dof_addr].copy())
            return (np.asarray(centers, dtype=np.float64),
                    np.asarray(jacobians, dtype=np.float64))

        return self._with_q(q, compute)

    def residuals(self, q, include_box=True, allow_box_contact_geometry=False):
        centers = self.sphere_centers(q)
        values = []

        for center, sphere in zip(centers, self.spheres):
            if not sphere.box_contact_geometry:
                values.append(center[2] - sphere.radius - self.d_ground)

            for obstacle_sdf in self.static_obstacles:
                values.append(obstacle_sdf(center) - sphere.radius - self.d_safe)

            skip_box = sphere.box_contact_geometry and allow_box_contact_geometry
            if self.box_sdf is not None and include_box and not skip_box:
                values.append(self.box_sdf(center) - sphere.radius - self.d_box)

        return np.asarray(values, dtype=np.float64)

    def residuals_and_jacobian(self, q, include_box=True,
                               allow_box_contact_geometry=False):
        centers, center_jacs = self.sphere_centers_and_jacobians(q)
        values = []
        jac_rows = []

        for center, center_jac, sphere in zip(
                centers, center_jacs, self.spheres):
            if not sphere.box_contact_geometry:
                values.append(center[2] - sphere.radius - self.d_ground)
                jac_rows.append(center_jac[2])

            for obstacle_sdf in self.static_obstacles:
                dist, grad = _sdf_value_and_gradient(obstacle_sdf, center)
                values.append(dist - sphere.radius - self.d_safe)
                jac_rows.append(grad @ center_jac)

            skip_box = sphere.box_contact_geometry and allow_box_contact_geometry
            if self.box_sdf is not None and include_box and not skip_box:
                dist, grad = _sdf_value_and_gradient(self.box_sdf, center)
                values.append(dist - sphere.radius - self.d_box)
                jac_rows.append(grad @ center_jac)

        return (np.asarray(values, dtype=np.float64),
                np.asarray(jac_rows, dtype=np.float64))

    def residuals_for_trajectory(self, qs, box_active_mask=None,
                                 box_contact_allowed_mask=None):
        qs = np.asarray(qs, dtype=np.float64)
        if box_active_mask is None:
            box_active_mask = np.ones(qs.shape[0], dtype=bool)
        box_active_mask = np.asarray(box_active_mask, dtype=bool)
        if box_active_mask.shape != (qs.shape[0],):
            raise ValueError("box_active_mask must match trajectory length")
        if box_contact_allowed_mask is None:
            box_contact_allowed_mask = np.zeros(qs.shape[0], dtype=bool)
        box_contact_allowed_mask = np.asarray(
            box_contact_allowed_mask, dtype=bool)
        if box_contact_allowed_mask.shape != (qs.shape[0],):
            raise ValueError(
                "box_contact_allowed_mask must match trajectory length")

        residuals = [
            self.residuals(
                q,
                include_box=bool(box_active_mask[k]),
                allow_box_contact_geometry=bool(box_contact_allowed_mask[k]),
            )
            for k, q in enumerate(qs)
        ]
        return np.concatenate(residuals) if residuals else np.zeros(0)


def default_ur10e_collision_spheres():
    """Conservative UR10e-style sphere cover in local body frames.

    The numbers are intentionally simple and slightly inflated for safety in
    this table-top setup. They cover arm links plus gripper geometry, while
    gripper-contact spheres are marked so box avoidance can be disabled during
    grasp/contact phases.
    """
    specs = [
        CollisionSphere("shoulder_link", [0.0, 0.0, 0.0], 0.115,
                        "shoulder"),

        CollisionSphere("upper_arm_link", [-0.10, 0.0, 0.02], 0.095,
                        "upper_arm_0"),
        CollisionSphere("upper_arm_link", [-0.28, 0.0, 0.02], 0.095,
                        "upper_arm_1"),
        CollisionSphere("upper_arm_link", [-0.46, 0.0, 0.02], 0.095,
                        "upper_arm_2"),
        CollisionSphere("upper_arm_link", [-0.60, 0.0, 0.02], 0.090,
                        "upper_arm_3"),

        CollisionSphere("forearm_link", [-0.10, 0.0, 0.03], 0.080,
                        "forearm_0"),
        CollisionSphere("forearm_link", [-0.27, 0.0, 0.08], 0.080,
                        "forearm_1"),
        CollisionSphere("forearm_link", [-0.44, 0.0, 0.13], 0.080,
                        "forearm_2"),
        CollisionSphere("forearm_link", [-0.56, 0.0, 0.17], 0.075,
                        "forearm_3"),

        CollisionSphere("wrist_1_link", [0.0, -0.06, 0.0], 0.075,
                        "wrist_1"),
        CollisionSphere("wrist_2_link", [0.0, 0.05, 0.0], 0.070,
                        "wrist_2"),
        CollisionSphere("wrist_3_link", [0.0, 0.0, 0.05], 0.065,
                        "wrist_3_0"),
        CollisionSphere("wrist_3_link", [0.0, 0.0, 0.14], 0.060,
                        "wrist_3_1"),

        CollisionSphere("ee_tcp", [0.0, 0.0, 0.0], 0.055,
                        "tool_tcp", box_contact_geometry=True),
        CollisionSphere("jaws_link", [0.0, 0.0, 0.04], 0.070,
                        "jaws", box_contact_geometry=True),
    ]
    return specs


def obstacle_sdfs_from_environment(env, excluded_obstacle_names=None):
    excluded = set(excluded_obstacle_names or [])
    sdfs = []
    for spec in env._obstacle_defs:
        if spec.get("name") in excluded:
            continue
        if spec.get("type") != "cylinder":
            continue
        radius, half_height = spec["size"]
        sdfs.append(FiniteCylinderSDF(spec["pos"], radius, half_height))
    return sdfs


def default_table_box_sdf(center=(0.35, 0.70, 0.14),
                          half_extents=(0.055, 0.055, 0.08)):
    return AxisAlignedBoxSDF(center, half_extents)


def make_default_ur10e_collision_model(env, arm,
                                       include_box=False,
                                       box_sdf=None,
                                       d_ground=0.02,
                                       d_safe=0.04,
                                       d_box=0.03,
                                       excluded_obstacle_names=None):
    return CollisionSphereModel(
        arm.model,
        arm.data,
        arm.qpos_addr,
        default_ur10e_collision_spheres(),
        static_obstacles=obstacle_sdfs_from_environment(
            env,
            excluded_obstacle_names=excluded_obstacle_names,
        ),
        box_sdf=(box_sdf if include_box else None),
        d_ground=d_ground,
        d_safe=d_safe,
        d_box=d_box,
        dof_addr=arm.dof_addr,
    )
