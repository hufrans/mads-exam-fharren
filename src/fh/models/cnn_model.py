import torch
import torch.nn as nn
from typing import Dict, Any
from loguru import logger # Importeer loguru's logger

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) model.
    Assumes input is 1D for simplicity, can be adapted for 2D images.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the CNNModel.

        Args:
            config (Dict[str, Any]): A dictionary containing model configuration parameters,
                                    e.g., 'input_channels', 'hidden_size', 'output_size',
                                    'conv_filters', 'kernel_size', 'use_dropout'.
        """
        super().__init__()
        logger.info("Initializing CNNModel...")
        if not isinstance(config, dict):
            logger.error(f"Invalid config type provided: {type(config)}. Expected dictionary.")
            raise TypeError("Config must be a dictionary.")

        # --- Validate and extract config parameters with logging ---
        input_channels = config.get("input_channels")
        output_size = config.get("output_size")
        conv_filters = config.get("conv_filters")
        kernel_size = config.get("kernel_size")
        hidden_size = config.get("hidden_size")
        input_size_after_flattening = config.get("input_size_after_flattening")

        if input_channels is None or not isinstance(input_channels, int) or input_channels <= 0:
            logger.error(f"Invalid or missing 'input_channels' in config: {input_channels}")
            raise ValueError("Config missing 'input_channels' or it's invalid.")
        if output_size is None or not isinstance(output_size, int) or output_size <= 0:
            logger.error(f"Invalid or missing 'output_size' in config: {output_size}")
            raise ValueError("Config missing 'output_size' or it's invalid.")
        if conv_filters is None or not isinstance(conv_filters, list) or not all(isinstance(f, int) and f > 0 for f in conv_filters):
            logger.error(f"Invalid or missing 'conv_filters' in config: {conv_filters}. Must be a list of positive integers.")
            raise ValueError("Config missing 'conv_filters' or it's invalid (must be a list of positive integers).")
        if kernel_size is None or not isinstance(kernel_size, int) or kernel_size <= 0:
            logger.error(f"Invalid or missing 'kernel_size' in config: {kernel_size}")
            raise ValueError("Config missing 'kernel_size' or it's invalid.")
        if hidden_size is None or not isinstance(hidden_size, int) or hidden_size <= 0:
            logger.error(f"Invalid or missing 'hidden_size' in config: {hidden_size}")
            raise ValueError("Config missing 'hidden_size' or it's invalid.")
        if input_size_after_flattening is None or not isinstance(input_size_after_flattening, int) or input_size_after_flattening <= 0:
            logger.error(f"Invalid or missing 'input_size_after_flattening' for CNN's linear layer: {input_size_after_flattening}. This needs to be calculated based on your data and conv layers.")
            raise ValueError("Config missing 'input_size_after_flattening' for CNN's linear layer, or it's invalid. This needs to be calculated based on your data and conv layers.")

        self.use_dropout = config.get("use_dropout", False)
        logger.debug(f"CNNModel config: input_channels={input_channels}, output_size={output_size}, conv_filters={conv_filters}, kernel_size={kernel_size}, hidden_size={hidden_size}, use_dropout={self.use_dropout}")

        # --- Convolutional Layers ---
        layers = []
        in_channels = input_channels
        for i, filters in enumerate(conv_filters):
            layers.append(nn.Conv1d(in_channels, filters, kernel_size=kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2)) # Reduce dimensionality
            logger.debug(f"Added Conv1d layer {i+1}: in_channels={in_channels}, out_channels={filters}, kernel_size={kernel_size}")
            in_channels = filters
        self.conv_layers = nn.Sequential(*layers)
        logger.debug("Convolutional layers built.")

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(input_size_after_flattening, hidden_size)
        self.relu = nn.ReLU()
        if self.use_dropout:
            dropout_rate = config.get("dropout_rate", 0.5)
            self.dropout = nn.Dropout(dropout_rate)
            logger.debug(f"Dropout layer added with rate: {dropout_rate}")
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        logger.info(f"CNNModel initialized. Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the CNN model.

        Args:
            x (torch.Tensor): The input tensor. Expected shape: (batch_size, input_channels, sequence_length).
                              If x is 2D (batch_size, sequence_length), it will be unsqueezed to add a channel dimension.

        Returns:
            torch.Tensor: The output tensor (logits).
        """
        if not isinstance(x, torch.Tensor):
            logger.error(f"Input to forward pass is not a torch.Tensor: {type(x)}")
            raise TypeError("Input must be a torch.Tensor.")
        
        logger.debug(f"CNNModel - Initial input tensor shape: {x.shape}")

        # Ensure input is 3D for Conv1d (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dimension if not present (e.g., (batch_size, sequence_length) -> (batch_size, 1, sequence_length))
            logger.debug(f"CNNModel - Input unsqueezed to 3D. New shape: {x.shape}")
        
        logger.debug(f"CNNModel - Input shape to conv_layers: {x.shape}") 
        x = self.conv_layers(x)
        logger.debug(f"CNNModel - After conv_layers shape: {x.shape}")

        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        logger.debug(f"CNNModel - After flatten shape: {x.shape}")
        
        # Defensive check for the flattened size to match the linear layer's input size
        if x.size(1) != self.fc1.in_features:
            logger.critical(f"Flattened input size {x.size(1)} does not match expected linear layer input size {self.fc1.in_features}. "
                            "Adjust 'input_size_after_flattening' in config or your model's architecture.")
            raise ValueError(f"Flattened input size {x.size(1)} does not match expected linear layer input size {self.fc1.in_features}. "
                             "Adjust 'input_size_after_flattening' in config or your model's architecture.")

        x = self.fc1(x)
        logger.debug(f"CNNModel - After fc1 shape: {x.shape}")
        x = self.relu(x)
        logger.debug(f"CNNModel - After ReLU shape: {x.shape}")
        if self.use_dropout:
            x = self.dropout(x)
            logger.debug(f"CNNModel - After Dropout shape: {x.shape}")
        x = self.fc2(x)
        logger.debug(f"CNNModel - After fc2 (output) shape: {x.shape}")
        return x