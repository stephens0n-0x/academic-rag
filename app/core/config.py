from pathlib import Path
import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config() -> dict:
    """Load and return application configuration from YAML."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()