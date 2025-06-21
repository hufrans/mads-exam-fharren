import torch
import torch.nn as nn
from typing import Dict, Any

class GRUModel(nn.Module):
    """
    A Gated Recurrent Unit (GRU) model.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the GRUModel.

        Args:
            config (Dict[str, Any]): A dictionary containing model configuration parameters,
                                    e.g., 'input_size', 'hidden_size', 'num_layers',
                                    'output_size', 'dropout'.
        """
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        if "input_size" not in config or not isinstance(config["input_size"], int) or config["input_size"] <= 0:
            raise ValueError("Config missing 'input_size' or it's invalid.")
        if "hidden_size" not in config or not isinstance(config["hidden_size"], int) or config["hidden_size"] <= 0:
            raise ValueError("Config missing 'hidden_size' or it's invalid.")
        if "num_layers" not in config or not isinstance(config["num_layers"], int) or config["num_layers"] <= 0:
            raise ValueError("Config missing 'num_layers' or it's invalid.")
        if "output_size" not in config or not isinstance(config["output_size"], int) or config["output_size"] <= 0:
            raise ValueError("Config missing 'output_size' or it's invalid.")

        self.gru = nn.GRU(
            config["input_size"],
            config["hidden_size"],
            config["num_layers"],
            batch_first=True,
            dropout=config.get("dropout", 0.0) if config["num_layers"] > 1 else 0.0 # Dropout only if num_layers > 1
        )
        self.fc = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the GRU model.

        Args:
            x (torch.Tensor): The input tensor. Expected shape: (batch_size, sequence_length, input_size).
                              If x is 2D (batch_size, input_size), it will be unsqueezed to add a sequence_length dimension.

        Returns:
            torch.Tensor: The output tensor (logits).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        
        # Ensure input is 3D for GRU (batch_size, sequence_length, input_size)
        # If your data is tabular (e.g., N_SAMPLES x N_FEATURES) and you want to treat it as a sequence of length 1,
        # then add a sequence dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add sequence_length dimension (e.g., (batch_size, input_size) -> (batch_size, 1, input_size))

        gru_out, _ = self.gru(x)
        # Take the output of the last time step
        output = self.fc(gru_out[:, -1, :])
        return output