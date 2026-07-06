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

# Externally chosen cylinder radii, in meters. The PDDL gives x/y centers and
# total z height; the radius is a scene-design choice.
PDDL_CYLINDER_RADII = {
    "in-2": 0.050,
    "in-4": 0.055,
    "in-5": 0.047,
    "out-2": 0.060,
}

PDDL_CYLINDER_COLORS = {
    "in-2":  [0.95, 0.68, 0.20, 0.85],
    "in-4":  [0.20, 0.75, 0.90, 0.85],
    "in-5":  [0.95, 0.45, 0.20, 0.85],
    "out-2": [0.88, 0.88, 0.24, 0.85],
}

REMOVED_PDDL_CYLINDERS = {"in-1", "in-3", "out-1"}

PDDL_LAYOUT_ROTATION_DEG = -90.0

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


def load_pddl_cylinder_obstacles(problem_path=PDDL_PROBLEM_PATH):
    """Build static cylinder obstacles from PDDL bucket coordinates."""
    obstacles = []
    bucket_coords = _read_pddl_bucket_xyz(problem_path)
    center_x, center_y = _bucket_layout_center(bucket_coords)

    for index, (name, (x, y, height)) in enumerate(bucket_coords.items()):
        if name in REMOVED_PDDL_CYLINDERS:
            continue
        if height <= 0.0:
            continue

        fallback_radius = 0.040 + 0.004 * (index % 6)
        radius = PDDL_CYLINDER_RADII.get(name, fallback_radius)
        half_height = 0.5 * height
        rotated_x, rotated_y = _rotate_xy(
            x, y, center_x, center_y, PDDL_LAYOUT_ROTATION_DEG)

        obstacles.append({
            "name": f"pddl_{name.replace('-', '_')}_cylinder",
            "pos": [float(rotated_x), float(rotated_y), half_height],
            "size": [radius, half_height],
            "rgba": PDDL_CYLINDER_COLORS.get(name, [0.9, 0.5, 0.1, 0.85]),
            "type": "cylinder",
            # MuJoCo cylinders have solid caps. Keep them visual-only here and
            # let the MPC lateral-surface SDF provide the obstacle constraint.
            "contype": 0,
            "conaffinity": 0,
        })
    return obstacles


OBSTACLE_DEFAULTS = load_pddl_cylinder_obstacles()


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
    """Wrap the robot and static PDDL cylinders used by the MPC scene."""

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
