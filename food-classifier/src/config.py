"""Load configuration with paths relative to its YAML file, not the shell cwd."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "convnextv2_base.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key, value in config.get("paths", {}).items():
        if value:
            path = Path(value).expanduser()
            config["paths"][key] = str((config_path.parent / path).resolve())
    pretrained = config.get("model", {}).get("pretrained_checkpoint")
    if pretrained:
        config["model"]["pretrained_checkpoint"] = str(
            (config_path.parent / Path(pretrained).expanduser()).resolve()
        )
    return config
