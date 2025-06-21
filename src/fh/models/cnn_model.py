import torch
import torch.nn as nn
from typing import Dict, Any

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
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        if "input_channels" not in config or not isinstance(config["input_channels"], int) or config["input_channels"] <= 0:
            raise ValueError("Config missing 'input_channels' or it's invalid.")
        if "output_size" not in config or not isinstance(config["output_size"], int) or config["output_size"] <= 0:
            raise ValueError("Config missing 'output_size' or it's invalid.")
        if "conv_filters" not in config or not isinstance(config["conv_filters"], list) or not all(isinstance(f, int) and f > 0 for f in config["conv_filters"]):
            raise ValueError("Config missing 'conv_filters' or it's invalid (must be a list of positive integers).")
        if "kernel_size" not in config or not isinstance(config["kernel_size"], int) or config["kernel_size"] <= 0:
            raise ValueError("Config missing 'kernel_size' or it's invalid.")

        self.use_dropout = config.get("use_dropout", False)

        layers = []
        in_channels = config["input_channels"]
        for filters in config["conv_filters"]:
            layers.append(nn.Conv1d(in_channels, filters, kernel_size=config["kernel_size"], padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2)) # Reduce dimensionality
            in_channels = filters
        self.conv_layers = nn.Sequential(*layers)

        # Calculate input size for the fully connected layer after convolutional layers
        # This requires a dummy forward pass or careful calculation based on input size and pooling
        # For simplicity, let's assume a fixed input size to the fully connected layer after some convolutions
        # In a real scenario, you'd calculate this dynamically.
        # Here's a placeholder, you'll need to adjust based on your actual input feature dimension.
        # Example: if input_size is 10, after a few conv and pool layers, it might be 1 or 2.
        # A more robust way is to pass a dummy tensor through conv_layers and check its size.
        
        # NOTE: Dummy input size for the fully connected layer.
        # If your data is tabular and you treat each feature as a "sequence element" in a single channel,
        # and your initial input_size is N_FEATURES, then after several Conv1d and MaxPool1d,
        # the size will be reduced. You MUST ensure this matches.
        # A common way is to instantiate a dummy input tensor and pass it through the conv layers:
        
        # Example for how to calculate input_size_after_conv dynamically (add this logic to main.py or here if needed)
        # Assuming input shape for Conv1d is (batch_size, input_channels, sequence_length)
        # where sequence_length is the original number of features.
        # dummy_input_len = config.get("original_input_features", 10) # Get this from the dataset's feature count
        # dummy_input_tensor = torch.randn(1, config["input_channels"], dummy_input_len)
        # dummy_output_conv = self.conv_layers(dummy_input_tensor)
        # fc_input_size = dummy_output_conv.view(dummy_output_conv.size(0), -1).size(1)
        # self.fc1 = nn.Linear(fc_input_size, config["hidden_size"])
        
        # For now, let's keep it simple and assume `input_size_after_conv` is a parameter
        # that needs to be configured correctly in the config or passed during instantiation.
        # Alternatively, if you have a fixed input size like 10 features, and two MaxPool1D(2) layers,
        # the effective sequence length after pooling would be 10 / (2*2) = 2.5, which typically floors to 2.
        # The `in_channels` after convolution is `config["conv_filters"][-1]`.
        # So, the linear layer input would be `config["conv_filters"][-1] * final_sequence_length`.
        
        # I'll use a safer, explicit approach assuming `input_size_after_flattening` can be provided
        # or calculated externally (e.g., in main.py) and passed.
        # For now, let's use a placeholder that needs to be calculated outside.
        # You'll need to pass 'input_size_after_flattening' from main.py after checking your data dimensions.
        if "input_size_after_flattening" not in config or not isinstance(config["input_size_after_flattening"], int) or config["input_size_after_flattening"] <= 0:
            raise ValueError("Config missing 'input_size_after_flattening' for CNN's linear layer, or it's invalid. This needs to be calculated based on your data and conv layers.")
            
        self.fc1 = nn.Linear(config["input_size_after_flattening"], config["hidden_size"])
        self.relu = nn.ReLU()
        if self.use_dropout:
            self.dropout = nn.Dropout(config.get("dropout_rate", 0.5))
        self.fc2 = nn.Linear(config["hidden_size"], config["output_size"])

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
            raise TypeError("Input must be a torch.Tensor.")
        
        # Ensure input is 3D for Conv1d (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dimension if not present (e.g., (batch_size, sequence_length) -> (batch_size, 1, sequence_length))
        
        print(f"DEBUG: Input shape to conv_layers: {x.shape}") # <-- VOEG DEZE REGEL TOE

        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        
        # Defensive check for the flattened size to match the linear layer's input size
        # This is where the 'input_size_after_flattening' becomes critical.
        if x.size(1) != self.fc1.in_features:
            raise ValueError(f"Flattened input size {x.size(1)} does not match expected linear layer input size {self.fc1.in_features}. "
                             "Adjust 'input_size_after_flattening' in config or your model's architecture.")

        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x