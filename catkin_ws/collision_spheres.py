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


@dataclass(frozen=True)
class StaticObstacleConstraint:
    sdf: object
    safety_margin: float
    name: str = ""


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
        self.d_ground = float(d_ground)
        self.d_safe = float(d_safe)
        self.d_box = float(d_box)
        self.static_obstacles = [
            self._normalize_static_obstacle(obstacle)
            for obstacle in (static_obstacles or [])
        ]
        self.box_sdf = box_sdf

        self.body_ids = []
        for sphere in self.spheres:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, sphere.body_name)
            if body_id < 0:
                raise ValueError(f"unknown body {sphere.body_name!r}")
            self.body_ids.append(body_id)

    def _normalize_static_obstacle(self, obstacle):
        if isinstance(obstacle, StaticObstacleConstraint):
            return obstacle
        return StaticObstacleConstraint(
            sdf=obstacle,
            safety_margin=self.d_safe,
        )

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

            for obstacle in self.static_obstacles:
                values.append(
                    obstacle.sdf(center)
                    - sphere.radius
                    - obstacle.safety_margin
                )

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

            for obstacle in self.static_obstacles:
                dist, grad = _sdf_value_and_gradient(obstacle.sdf, center)
                values.append(dist - sphere.radius - obstacle.safety_margin)
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
    ]
    return specs


def obstacle_sdfs_from_environment(env, target_obstacle_name=None,
                                   target_d_safe_factor=0.5,
                                   d_safe=0.04):
    return []


def default_table_box_sdf(center=(0.35, 0.70, 0.14),
                          half_extents=(0.055, 0.055, 0.08)):
    return AxisAlignedBoxSDF(center, half_extents)


def make_default_ur10e_collision_model(env, arm,
                                       include_box=False,
                                       box_sdf=None,
                                       d_ground=0.02,
                                       d_safe=0.04,
                                       d_box=0.03,
                                       target_obstacle_name=None,
                                       target_d_safe_factor=0.5):
    return CollisionSphereModel(
        arm.model,
        arm.data,
        arm.qpos_addr,
        default_ur10e_collision_spheres(),
        static_obstacles=obstacle_sdfs_from_environment(
            env,
            target_obstacle_name=target_obstacle_name,
            target_d_safe_factor=target_d_safe_factor,
            d_safe=d_safe,
        ),
        box_sdf=(box_sdf if include_box else None),
        d_ground=d_ground,
        d_safe=d_safe,
        d_box=d_box,
        dof_addr=arm.dof_addr,
    )
