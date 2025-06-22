import torch
import torch.nn as nn
from typing import Dict, Any

class SqueezeExcitation(nn.Module):
    """
    Implementatie van het Squeeze-and-Excitation (SE) blok voor 1D Conv features.
    Dit blok kalibreert kanaal-wijze feature responses door expliciet rekening te houden
    met de onderlinge afhankelijkheden tussen kanalen.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialiseert het SqueezeExcitation blok.
        Args:
            config (Dict[str, Any]): Configuratie dictionary met 'channel' en 'reduction_ratio'.
        """
        # Foutieve 'SqueezeExcency' is gecorrigeerd naar 'SqueezeExcitation'
        super(SqueezeExcitation, self).__init__() 
        channel = config.get("channel")
        reduction = config.get("reduction_ratio", 16) # Standaard 16 als niet opgegeven

        if channel is None:
            raise ValueError("SqueezeExcitation: 'channel' moet aanwezig zijn in de configuratie.")
        if channel // reduction == 0:
            raise ValueError(f"SqueezeExcitation: Reduction ratio {reduction} is te groot voor {channel} kanalen. Kanaal na reductie zou 0 zijn.")

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Squeeze operatie (Global Average Pooling)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), # Reductie laag
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), # Expansie laag
            nn.Sigmoid() # Sigmoid om output tussen 0 en 1 te krijgen
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definieert de forward pass van het SqueezeExcitation blok.
        Args:
            x (torch.Tensor): De input tensor van vorm (batch_size, channels, length).
        Returns:
            torch.Tensor: De geschaalde output tensor.
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze: (batch_size, channels)
        y = self.fc(y).view(b, c, 1)    # Excitation: (batch_size, channels, 1)
        return x * y.expand_as(x)       # Scale: Vermenigvuldig met de input.

class CNNSESkipBlock(nn.Module):
    """
    Een bouwblok voor een CNN met Squeeze-and-Excitation en een skip-verbinding.
    Dit blok omvat een 1D convolutionele laag, ReLU activatie, Max Pooling,
    een Squeeze-and-Excitation blok, en een skip-verbinding.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialiseert een CNNSESkipBlock.
        Args:
            config (Dict[str, Any]): Configuratie dictionary met 'in_channels', 'out_channels', 'kernel_size_per_block', 'reduction_ratio'.
        """
        super(CNNSESkipBlock, self).__init__()
        in_channels = config.get("in_channels")
        out_channels = config.get("out_channels")
        kernel_size = config.get("kernel_size_per_block")
        reduction_ratio = config.get("reduction_ratio")

        if any(p is None for p in [in_channels, out_channels, kernel_size, reduction_ratio]):
            raise ValueError("CNNSESkipBlock: Alle verplichte parameters (in_channels, out_channels, kernel_size_per_block, reduction_ratio) moeten in de configuratie aanwezig zijn.")

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # Halveert de lengte
        
        # Geef de config voor SqueezeExcitation door
        se_config = {"channel": out_channels, "reduction_ratio": reduction_ratio}
        self.se = SqueezeExcitation(se_config)

        # Skip-verbinding: Als in_channels niet gelijk is aan out_channels,
        # moet de skip-verbinding een 1x1 convolutie gebruiken om de kanalen te matchen.
        # De output van de skip-conv moet ook door de pooling.
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definieert de forward pass van de CNNSESkipBlock.
        Args:
            x (torch.Tensor): De input tensor van vorm (batch_size, channels, length).
        Returns:
            torch.Tensor: De output tensor.
        """
        # Pas de skip-verbinding toe en vervolgens pooling
        identity = self.pool(self.skip_conv(x)) 
        
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.se(out)

        # Voeg de skip-verbinding toe (element-wise optellen)
        return out + identity

class CNNSESkipModel(nn.Module):
    """
    Een Convolutioneel Neuraal Netwerk met Squeeze-and-Excitation blokken en skip-verbindingen.
    Deze architectuur stapelt meerdere CNNSESkipBlock-lagen gevolgd door
    volledig verbonden lagen voor classificatie. Geschikt voor 1D numerieke features.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialiseert de CNNSESkipModel.
        Args:
            config (Dict[str, Any]): Configuratie dictionary met alle modelparameters.
        """
        super(CNNSESkipModel, self).__init__()
        
        input_channels = config.get("input_channels")
        num_features = config.get("num_features")
        hidden_size = config.get("hidden_size")
        output_size = config.get("output_size")
        num_layers = config.get("num_layers")
        conv_filters_per_block = config.get("conv_filters_per_block")
        kernel_size_per_block = config.get("kernel_size_per_block")
        reduction_ratio = config.get("reduction_ratio")
        use_dropout = config.get("use_dropout", False)
        dropout_rate = config.get("dropout_rate", 0.5)
        input_size_after_flattening = config.get("input_size_after_flattening")

        if any(p is None for p in [input_channels, num_features, hidden_size, output_size, num_layers, conv_filters_per_block, kernel_size_per_block, reduction_ratio, input_size_after_flattening]):
            raise ValueError("CNNSESkipModel: Alle verplichte parameters ontbreken in de configuratie.")

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        layers = []
        in_c = input_channels
        for _ in range(num_layers):
            block_config = {
                "in_channels": in_c,
                "out_channels": conv_filters_per_block,
                "kernel_size_per_block": kernel_size_per_block,
                "reduction_ratio": reduction_ratio
            }
            layers.append(CNNSESkipBlock(block_config))
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            in_c = conv_filters_per_block # Output kanalen van vorig blok worden input kanalen van volgend blok
        
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(input_size_after_flattening, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate) if self.use_dropout else nn.Identity(),
            nn.Linear(hidden_size, output_size)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definieert de forward pass van de CNNSESkipModel.
        Args:
            x (torch.Tensor): De input tensor van vorm (batch_size, num_features).
        Returns:
            torch.Tensor: De output tensor na de forward pass.
        """
        x = x.unsqueeze(1) # Voeg een kanaal dimensie toe: (batch_size, 1, num_features)
        
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten alle dimensies behalve de batch dimensie
        
        x = self.classifier(x)
        return self.softmax(x)
