import torch
import torch.nn as nn
from typing import Dict, Any

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
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        if "input_size" not in config or not isinstance(config["input_size"], int) or config["input_size"] <= 0:
            raise ValueError("Config missing 'input_size' or it's invalid.")
        if "output_size" not in config or not isinstance(config["output_size"], int) or config["output_size"] <= 0:
            raise ValueError("Config missing 'output_size' or it's invalid.")

        self.fc1 = nn.Linear(config["input_size"], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, config["output_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor (logits).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x