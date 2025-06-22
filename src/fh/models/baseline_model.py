import torch
import torch.nn as nn
from typing import Dict, Any
from loguru import logger # Importeer loguru's logger

class BaselineModel(nn.Module):
    """
    A simple baseline model, e.g., a basic Multi-Layer Perceptron (MLP).
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the BaselineModel.

        Args:
            config (Dict[str, Any]): A dictionary containing model configuration parameters,
                                    e.g., 'input_size', 'output_size'.
        """
        super().__init__()
        logger.info("Initializing BaselineModel...")
        if not isinstance(config, dict):
            logger.error(f"Invalid config type provided: {type(config)}. Expected dictionary.")
            raise TypeError("Config must be a dictionary.")
        
        input_size = config.get("input_size")
        output_size = config.get("output_size")

        if input_size is None or not isinstance(input_size, int) or input_size <= 0:
            logger.error(f"Invalid or missing 'input_size' in config: {input_size}")
            raise ValueError("Config missing 'input_size' or it's invalid.")
        if output_size is None or not isinstance(output_size, int) or output_size <= 0:
            logger.error(f"Invalid or missing 'output_size' in config: {output_size}")
            raise ValueError("Config missing 'output_size' or it's invalid.")

        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        
        logger.info(f"BaselineModel initialized with input_size={input_size}, hidden_size=64, output_size={output_size}.")
        logger.debug(f"Model layers: fc1={self.fc1}, relu={self.relu}, fc2={self.fc2}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor (logits).
        """
        if not isinstance(x, torch.Tensor):
            logger.error(f"Input to forward pass is not a torch.Tensor: {type(x)}")
            raise TypeError("Input must be a torch.Tensor.")
        
        logger.debug(f"BaselineModel - Input tensor shape: {x.shape}")
        x = self.fc1(x)
        logger.debug(f"BaselineModel - After fc1 shape: {x.shape}")
        x = self.relu(x)
        logger.debug(f"BaselineModel - After ReLU shape: {x.shape}")
        x = self.fc2(x)
        logger.debug(f"BaselineModel - After fc2 (output) shape: {x.shape}")
        return x