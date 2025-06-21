import toml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a TOML file.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration.
    """
    try:
        config = toml.load(config_path)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading TOML config: {e}")