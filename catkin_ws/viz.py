#!/usr/bin/env python3
"""
Shared visualization helpers built on top of mujoco.viewer.user_scn.

All helpers append to viewer.user_scn; the caller is responsible for
setting `viewer.user_scn.ngeom` to the total number of geoms used (or
relying on `draw_scene` which manages that automatically).
"""
import numpy as np
import mujoco


def _init_capsule(geom, rgba):
    """Initialize a user-scene geom as a capsule (size set by mjv_connector)."""
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float64),
    )


def _connect(geom, a, b, width):
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )


def draw_polyline(viewer, start_idx, points, rgba, width=0.003):
    """Render a polyline starting at user_scn.geoms[start_idx].

    Returns the next free geom index. Caller updates viewer.user_scn.ngeom.
    """
    pts = list(points)
    idx = start_idx
    for i in range(len(pts) - 1):
        geom = viewer.user_scn.geoms[idx]
        _init_capsule(geom, rgba)
        _connect(geom, pts[i], pts[i + 1], width)
        idx += 1
    return idx


def draw_sphere_marker(viewer, start_idx, position,
                       rgba=(0.0, 0.1, 1.0, 1.0), radius=0.018):
    """Render one sphere marker and return the next free geom index."""
    geom = viewer.user_scn.geoms[start_idx]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float64),
    )
    return start_idx + 1
