#!/usr/bin/env python3
import numpy as np
import mujoco

from robot_config import get_config


class Robot:
    """Minimal MuJoCo robot wrapper required by the MPC demo."""

    def __init__(self, robot="ur10e", model=None):
        cfg = get_config(robot)
        self._robot_name  = robot
        self._ee_body     = cfg["ee_body"]
        self.n            = cfg["n_arm_joints"]
        self._home        = np.array(cfg.get("home_qpos", [0.0] * self.n))
        v = cfg.get("vel_limits");  self._vel_limits = np.array(v) if v else None

        self._model = model if model is not None else mujoco.MjModel.from_xml_path(cfg["xml"])
        self._data  = mujoco.MjData(self._model)
        self._ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._ee_body)

        self.reset()

    def reset(self):
        """Return to home position with zero velocity."""
        self._data.qpos[:self.n]    = self._home
        self._data.qvel[:]          = 0.0
        self._data.qfrc_applied[:]  = 0.0   # clear any torques from previous run
        mujoco.mj_forward(self._model, self._data)

    def step(self, tau=None):
        """Advance one simulation step. tau: joint torques (n,), None = gravity comp only."""
        self._data.qfrc_applied[:] = 0.0
        if tau is not None:
            self._data.qfrc_applied[:self.n] = tau
        else:
            self._data.qfrc_applied[:self.n] = self._data.qfrc_bias[:self.n]
        mujoco.mj_step(self._model, self._data)

    @property
    def vel_limits(self):
        """Max joint velocities (n,) in rad/s. None if not specified in config."""
        return self._vel_limits

    @property
    def ee_pos(self):
        """EE position (3,) in world frame, meters."""
        return self._data.xpos[self._ee_id].copy()

    @property
    def ee_rot(self):
        """EE rotation matrix (3,3) in world frame."""
        return self._data.xmat[self._ee_id].reshape(3, 3).copy()

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data
