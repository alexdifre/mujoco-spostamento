#!/usr/bin/env python3
"""Build the UR10e MuJoCo scene used by the MPC demo."""
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from robot import Robot

PDDL_PROBLEM_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "problem_chem_simplified_no_redundant_predicates.pddl",
))

PDDL_LAYOUT_ROTATION_DEG = -90.0

TABLE_TOP_VERTICES = {
    "v2": (-0.426518,  1.232840, 0.002500),
    "v4": (-0.427621, -0.152159, 0.002500),
    "v6": ( 0.427379, -0.152840, 0.002500),
    "v8": ( 0.428481,  1.232159, 0.002500),
}
TABLE_X_MIN = min(v[0] for v in TABLE_TOP_VERTICES.values())
TABLE_X_MAX = max(v[0] for v in TABLE_TOP_VERTICES.values())
TABLE_Y_MIN = min(v[1] for v in TABLE_TOP_VERTICES.values())
TABLE_Y_MAX = max(v[1] for v in TABLE_TOP_VERTICES.values())
TABLE_TOP_Z = 0.002500

# Five close cubes on the table. MuJoCo box geoms use half-extents, so the
# z-position is the cube center and the bottom face sits on TABLE_TOP_Z.
TABLE_CUBE_SIDE = 0.050
TABLE_CUBE_CENTERS_XY = {
    "cube_1": (-0.070, 0.640),
    "cube_2": ( 0.025, 0.640),
    "cube_3": ( 0.120, 0.640),
    "cube_4": (-0.025, 0.735),
    "cube_5": ( 0.070, 0.735),
}
TABLE_CUBE_COLORS = {
    "cube_1": [0.90, 0.20, 0.18, 1.0],
    "cube_2": [0.16, 0.54, 0.92, 1.0],
    "cube_3": [0.18, 0.72, 0.34, 1.0],
    "cube_4": [0.95, 0.73, 0.20, 1.0],
    "cube_5": [0.62, 0.34, 0.88, 1.0],
}

_PDDL_BUCKET_RE = re.compile(
    r"\(=\s+\(bucket-([xyz])\s+([^\s)]+)\)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def _read_pddl_bucket_xyz(problem_path):
    """Return {bucket_name: (x, y, z_height)} parsed from a PDDL problem."""
    try:
        with open(problem_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}

    coords = {}
    for axis, name, value in _PDDL_BUCKET_RE.findall(text):
        coords.setdefault(name, {})[axis] = float(value)

    return {
        name: (values["x"], values["y"], values["z"])
        for name, values in coords.items()
        if {"x", "y", "z"}.issubset(values)
    }


def _rotate_xy(x, y, cx, cy, degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def _bucket_layout_center(bucket_coords):
    if not bucket_coords:
        return 0.0, 0.0
    center_x = sum(x for x, _, _ in bucket_coords.values()) / len(bucket_coords)
    center_y = sum(y for _, y, _ in bucket_coords.values()) / len(bucket_coords)
    return center_x, center_y


def load_pddl_bucket_targets(problem_path=PDDL_PROBLEM_PATH,
                             rotate_layout=True,
                             z_clearance=0.0):
    """Return scene-frame EE targets for PDDL beakers keyed by beaker name."""
    bucket_coords = _read_pddl_bucket_xyz(problem_path)
    center_x, center_y = _bucket_layout_center(bucket_coords)

    targets = {}
    for name, (x, y, z) in bucket_coords.items():
        if rotate_layout:
            x, y = _rotate_xy(x, y, center_x, center_y,
                              PDDL_LAYOUT_ROTATION_DEG)
        targets[name] = np.array(
            [float(x), float(y), float(z) + float(z_clearance)],
            dtype=np.float64,
        )
    return targets


def load_table_cube_obstacles():
    """Build the default static cube layout on the provided table surface."""
    half_side = 0.5 * TABLE_CUBE_SIDE
    z_center = TABLE_TOP_Z + half_side
    obstacles = []

    for name, (x, y) in TABLE_CUBE_CENTERS_XY.items():
        if not (TABLE_X_MIN + half_side <= x <= TABLE_X_MAX - half_side):
            raise ValueError(f"{name} x center leaves the table bounds")
        if not (TABLE_Y_MIN + half_side <= y <= TABLE_Y_MAX - half_side):
            raise ValueError(f"{name} y center leaves the table bounds")

        obstacles.append({
            "name": name,
            "pos": [float(x), float(y), float(z_center)],
            "size": [half_side, half_side, half_side],
            "rgba": TABLE_CUBE_COLORS.get(name, [0.8, 0.3, 0.2, 1.0]),
            "type": "box",
        })

    return obstacles


OBSTACLE_DEFAULTS = load_table_cube_obstacles()


def _fmt(vec):
    return " ".join(repr(float(v)) for v in vec)


def _add_static_obstacle(worldbody, spec):
    """Append a static body (no joint) attached to the world."""
    name  = spec["name"]
    pos   = spec["pos"]
    shape = spec.get("type", "box")
    size  = spec["size"]
    rgba  = spec.get("rgba", [0.9, 0.5, 0.1, 1.0])

    b = ET.SubElement(worldbody, 'body', name=name, pos=_fmt(pos))
    geom_attrs = {
        "name": f"{name}_geom",
        "type": shape,
        "size": _fmt(size),
        "rgba": _fmt(rgba),
    }
    for key in ("contype", "conaffinity", "condim", "margin", "gap"):
        if key in spec:
            geom_attrs[key] = str(spec[key])
    ET.SubElement(b, 'geom', **geom_attrs)


# ── Scene builder ──────────────────────────────────────────────────────────────

def _build_model(robot_xml_path, obstacles):
    """Load the robot XML, add scene content, compile, and return MjModel."""
    abs_xml  = os.path.abspath(robot_xml_path)
    base_dir = os.path.dirname(abs_xml)

    tree = ET.parse(abs_xml)
    root = tree.getroot()

    # mj_compile loses the original file path, so relative mesh paths fail
    # unless meshdir is rewritten to an absolute one.
    compiler = root.find('compiler')
    rel_meshdir = compiler.get('meshdir', '')
    abs_meshdir = os.path.join(base_dir, rel_meshdir) if rel_meshdir else base_dir
    compiler.set('meshdir', abs_meshdir)

    worldbody = root.find('worldbody')

    for spec in obstacles:
        _add_static_obstacle(worldbody, spec)

    xml_str = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml_str)


# ── Environment class ──────────────────────────────────────────────────────────

class environment:
    """Wrap the robot and static table-top obstacles used by the MPC scene."""

    def __init__(self, robot="ur10e", obstacles=None):
        from robot_config import get_config
        cfg = get_config(robot)

        self._obstacle_defs = list(obstacles if obstacles is not None else OBSTACLE_DEFAULTS)

        model = _build_model(cfg["xml"], self._obstacle_defs)
        self.robot = Robot(robot, model=model)

    def reset(self):
        """Reset arm to home."""
        self.robot.reset()
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def step(self, tau=None):
        """Advance one simulation step (delegates to robot.step)."""
        self.robot.step(tau)
