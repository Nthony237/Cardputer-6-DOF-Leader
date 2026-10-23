"""End-to-end: start the sim teleop headless (no keyboard) for a moment and quit cleanly."""
import socket

import pytest

pytest.importorskip("viser")
pytest.importorskip("yourdfpy")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sim_runs_and_exits():
    from cardputer_teleop.teleop import main

    main(["--mode", "sim", "--keys", "none", "--duration", "0.5", "--port", str(free_port())])


def test_config_file(tmp_path):
    from cardputer_teleop.config import PLACEHOLDER_URDF, load_config

    p = tmp_path / "yam.toml"
    p.write_text(
        f'[robot]\nurdf = "{PLACEHOLDER_URDF}"\nee_link = "link6"\n'
        "[workspace]\nlower = [0.0, -0.1, 0.0]\n[speed]\nlinear = 0.05\n"
    )
    cfg = load_config(p)
    assert cfg.robot.ee_link == "link6"
    assert cfg.workspace.lower.tolist() == [0.0, -0.1, 0.0]
    assert cfg.speed.linear == 0.05


def test_example_config_loads_and_matches_chain():
    from pathlib import Path

    from cardputer_teleop.config import load_config
    from cardputer_teleop.kinematics import SerialChain

    cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
    chain = SerialChain(cfg.robot.urdf, cfg.robot.ee_link)
    assert len(cfg.robot.home) == chain.dof
    assert cfg.workspace.contains(chain.fk(cfg.robot.home)[:3, 3])
