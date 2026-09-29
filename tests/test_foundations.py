"""Exercise the installed package and offline development contract."""

import importlib.metadata
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pytest_socket import SocketBlockedError

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_has_expected_version() -> None:
    assert importlib.metadata.version("handoff") == "0.1.0"


def test_installed_package_imports_outside_checkout(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import handoff; print(handoff.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_config_composes_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name="run_eval")
    assert config.seed == 0
    assert isinstance(config.seed, int)


def test_network_is_disabled() -> None:
    with (
        pytest.warns(UserWarning, match="socket.socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket()
