"""viser browser view: robot URDF, TCP and target frames, workspace box, and teleop controls."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from cardputer_teleop.safety import Workspace
from cardputer_teleop.transforms import make_transform, matrix_to_wxyz, wxyz_to_matrix


class SimViewer:
    def __init__(self, urdf_path: str | Path, chain_joint_names: list[str], workspace: Workspace,
                 gripper_joint: str | None = None, port: int = 8080, linear_speed: float = 0.05,
                 max_linear_speed: float = 0.15):
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (0.9, -0.9, 0.7)
        self.server.initial_camera.look_at = (0.25, 0.0, 0.15)
        self.urdf = ViserUrdf(self.server, Path(urdf_path), root_node_name="/robot")
        self._names = list(self.urdf.get_actuated_joint_names())
        self._chain_idx = [self._names.index(n) for n in chain_joint_names]
        self._gripper_idx = self._names.index(gripper_joint) if gripper_joint in self._names else None
        self._gripper_range = self.urdf.get_actuated_joint_limits().get(gripper_joint, (0.0, 0.0))

        self.server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.1)
        center = (workspace.lower + workspace.upper) / 2
        self.server.scene.add_box(
            "/workspace", color=(80, 160, 255), dimensions=tuple(workspace.upper - workspace.lower),
            position=tuple(center), wireframe=True,
        )
        self.tcp_frame = self.server.scene.add_frame("/tcp", axes_length=0.06, axes_radius=0.004)
        self.target = self.server.scene.add_transform_controls("/target", scale=0.12)
        self.drag_target: np.ndarray | None = None
        self._last_drag = -np.inf

        @self.target.on_update
        def _(event):
            # Only the browser triggers on_update, so every call is the user dragging.
            ended = getattr(event, "phase", None) == "end"
            self._last_drag = -np.inf if ended else time.monotonic()
            self.drag_target = make_transform(wxyz_to_matrix(self.target.wxyz), self.target.position)

        gui = self.server.gui
        self.status = gui.add_markdown("starting...")
        self.hold = gui.add_checkbox("Hold (space)", initial_value=False)
        self.frame = gui.add_dropdown("Rotate in frame (t)", ("base", "tool"), initial_value="tool")
        self.speed = gui.add_slider("Linear speed (m/s)", 0.01, max_linear_speed, 0.01, linear_speed)
        self.home_requested = False
        home = gui.add_button("Home (h)")

        @home.on_click
        def _(_):
            self.home_requested = True

    def take_drag_target(self) -> np.ndarray | None:
        T, self.drag_target = self.drag_target, None
        return T

    def update(self, q: np.ndarray, gripper01: float, T_tcp: np.ndarray, T_target: np.ndarray, status: str):
        cfg = np.zeros(len(self._names))
        cfg[self._chain_idx] = q
        if self._gripper_idx is not None:
            lo, hi = self._gripper_range
            cfg[self._gripper_idx] = lo + gripper01 * (hi - lo)
        self.urdf.update_cfg(cfg)
        self.tcp_frame.position = T_tcp[:3, 3]
        self.tcp_frame.wxyz = matrix_to_wxyz(T_tcp[:3, :3])
        if time.monotonic() - self._last_drag > 0.3:  # don't fight the user's drag
            self.target.position = T_target[:3, 3]
            self.target.wxyz = matrix_to_wxyz(T_target[:3, :3])
        self.status.content = status

    def close(self):
        self.server.stop()
