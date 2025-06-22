import toml
from typing import Dict, Any
from loguru import logger # Importeer loguru's logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a TOML file.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration.
    """
    logger.info(f"Attempting to load configuration from: {config_path}")
    try:
        config = toml.load(config_path)
        logger.success(f"Configuration successfully loaded from: {config_path}")
        logger.debug(f"Loaded config keys: {list(config.keys())}") # Debug-level detail
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except Exception as e:
        logger.error(f"Error loading TOML config from {config_path}: {e}")
        raise ValueError(f"Error loading TOML config: {e}")