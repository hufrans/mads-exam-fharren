# src/fh/model_selector.py
import torch.nn as nn
from typing import Dict, Any
from loguru import logger

# Importeer hier al je modeldefinities
from src.fh.models.baseline_model import BaselineModel
from src.fh.models.cnn_model import CNNModel
from src.fh.models.gru_model import GRUModel


def get_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    """
    Kiest en retourneert een model op basis van de opgegeven naam en configuratie.

    Args:
        model_name (str): De naam van het te laden model (bijv. "baseline_model", "cnn_model").
        model_config (Dict[str, Any]): Een dictionary met specifieke configuratieparameters
                                       voor het gekozen model, zoals input_size, hidden_size, etc.

    Returns:
        nn.Module: Een geïnitialiseerd PyTorch model.

    Raises:
        ValueError: Als een onbekende model_name wordt opgegeven.
    """
    logger.info(f"Model selector: aanmaken van model '{model_name}' met config: {model_config}")

    if model_name == "baseline_model":
        # Geef de volledige model_config dictionary door
        return BaselineModel(config=model_config) 
    elif model_name == "cnn_model":
        # Herhaal dit voor CNNModel als deze ook een 'config' dictionary verwacht
        return CNNModel(config=model_config)
    elif model_name == "gru_model":
        # Herhaal dit voor GRUModel als deze ook een 'config' dictionary verwacht
        return GRUModel(config=model_config)
    else:
        logger.error(f"Onbekend model '{model_name}' gevraagd.")
        raise ValueError(f"Onbekend model: {model_name}")