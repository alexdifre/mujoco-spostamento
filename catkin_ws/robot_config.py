#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET

MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

ROBOT_CONFIGS = {
    "ur10e": {
        "xml":       "ur10e/ur10e.xml",
        "ee_body":   "ee_tcp",
        "home_qpos": [1.57089435, -1.42615965, 1.58303971, -1.69997267, -1.57142825, -0.00000032],
        "vel_limits": [2.094, 2.094, 3.141, 3.141, 3.141, 3.141],
    },
}


def _parse_xml(xml_path):
    """Auto-detect the arm joint count from MuJoCo motor actuators."""
    tree = ET.parse(xml_path)
    actuator = tree.getroot().find("actuator")

    motor_count = 0
    for act in actuator:
        if act.tag == "motor":
            motor_count += 1

    return motor_count


def get_config(robot_name):
    cfg = ROBOT_CONFIGS[robot_name].copy()
    cfg["xml"] = os.path.join(MODELS_ROOT, cfg["xml"])

    cfg["n_arm_joints"] = _parse_xml(cfg["xml"])

    return cfg
