"""Where gr-brain keeps your API address and access token.

``GR_BRAIN_API`` / ``GR_BRAIN_TOKEN`` override the saved values (handy in CI);
``GR_BRAIN_CONFIG_DIR`` moves the config folder.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_API = "https://api.genuinerobotics.com"


def config_dir() -> Path:
    if os.environ.get("GR_BRAIN_CONFIG_DIR"):
        return Path(os.environ["GR_BRAIN_CONFIG_DIR"])
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "gr-brain"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "gr-brain"


@dataclass
class Config:
    api: str = DEFAULT_API
    token: Optional[str] = None


def load() -> Config:
    path = config_dir() / "config.json"
    cfg = Config()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg.api = data.get("api") or DEFAULT_API
            cfg.token = data.get("token") or None
        except (OSError, ValueError):
            pass
    cfg.api = (os.environ.get("GR_BRAIN_API") or cfg.api).rstrip("/")
    cfg.token = os.environ.get("GR_BRAIN_TOKEN") or cfg.token
    return cfg


def save(cfg: Config) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "config.json"
    path.write_text(json.dumps({"api": cfg.api, "token": cfg.token}, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        # The token is a credential: readable by you only.
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path
