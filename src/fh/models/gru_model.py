import torch
import torch.nn as nn
from typing import Dict, Any
from loguru import logger # Importeer loguru's logger

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
        logger.info("Initializing GRUModel...")
        if not isinstance(config, dict):
            logger.error(f"Invalid config type provided: {type(config)}. Expected dictionary.")
            raise TypeError("Config must be a dictionary.")

        # --- Validate and extract config parameters with logging ---
        input_size = config.get("input_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        output_size = config.get("output_size")
        dropout = config.get("dropout", 0.0)

        if input_size is None or not isinstance(input_size, int) or input_size <= 0:
            logger.error(f"Invalid or missing 'input_size' in config: {input_size}")
            raise ValueError("Config missing 'input_size' or it's invalid.")
        if hidden_size is None or not isinstance(hidden_size, int) or hidden_size <= 0:
            logger.error(f"Invalid or missing 'hidden_size' in config: {hidden_size}")
            raise ValueError("Config missing 'hidden_size' or it's invalid.")
        if num_layers is None or not isinstance(num_layers, int) or num_layers <= 0:
            logger.error(f"Invalid or missing 'num_layers' in config: {num_layers}")
            raise ValueError("Config missing 'num_layers' or it's invalid.")
        if output_size is None or not isinstance(output_size, int) or output_size <= 0:
            logger.error(f"Invalid or missing 'output_size' in config: {output_size}")
            raise ValueError("Config missing 'output_size' or it's invalid.")

        # Dropout only if num_layers > 1, as per PyTorch GRU documentation
        actual_dropout = dropout if num_layers > 1 else 0.0
        if num_layers == 1 and dropout > 0:
            logger.warning(f"Dropout ({dropout}) specified for GRU with num_layers=1. Setting dropout to 0.0 as per PyTorch documentation.")
        elif num_layers > 1 and dropout > 0:
            logger.debug(f"GRU dropout rate set to {dropout} for {num_layers} layers.")

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=actual_dropout
        )
        logger.debug(f"GRU layer initialized: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={actual_dropout}")

        self.fc = nn.Linear(hidden_size, output_size)
        logger.debug(f"Fully connected layer initialized: in_features={hidden_size}, out_features={output_size}")
        
        logger.info(f"GRUModel initialized. Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

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
            logger.error(f"Input to forward pass is not a torch.Tensor: {type(x)}")
            raise TypeError("Input must be a torch.Tensor.")
        
        logger.debug(f"GRUModel - Initial input tensor shape: {x.shape}")
        
        # Ensure input is 3D for GRU (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add sequence_length dimension (e.g., (batch_size, input_size) -> (batch_size, 1, input_size))
            logger.debug(f"GRUModel - Input unsqueezed to 3D. New shape: {x.shape}")
        
        gru_out, hidden = self.gru(x)
        logger.debug(f"GRUModel - After GRU layer, gru_out shape: {gru_out.shape}, hidden state shape: {hidden.shape}")

        # Take the output of the last time step for classification
        # gru_out[:, -1, :] selects all batches, the last sequence element, and all features from that element.
        output = self.fc(gru_out[:, -1, :])
        logger.debug(f"GRUModel - After FC layer (output) shape: {output.shape}")
        return output