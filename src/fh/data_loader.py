import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and preparing data from Parquet files.
    """
    def __init__(self, data_path: str, feature_columns: list[str], target_column: str) -> None:
        """
        Initializes the dataset.

        Args:
            data_path (str): The path to the Parquet data file.
            feature_columns (list[str]): A list of column names to be used as features.
            target_column (str): The name of the column to be used as the target.
        """
        try:
            # Lees Parquet-bestand in plaats van CSV
            self.data = pd.read_parquet(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        except Exception as e:
            raise IOError(f"Error loading data from {data_path}: {e}")

        if not all(col in self.data.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in self.data.columns]
            raise ValueError(f"Missing feature columns in data: {missing_cols}")
        if target_column not in self.data.columns:
            raise ValueError(f"Missing target column '{target_column}' in data.")

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and target.
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.long) # Assuming classification target
        return features, targets

def get_data_loaders(
    train_data_path: str,
    test_data_path: str,
    feature_columns: list[str],
    target_column: str,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for training and testing.

    Args:
        train_data_path (str): Path to the training data Parquet file.
        test_data_path (str): Path to the test data Parquet file.
        feature_columns (list[str]): Columns to use as features.
        target_column (str): Column to use as target.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the train and test DataLoaders.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    train_dataset = CustomDataset(train_data_path, feature_columns, target_column)
    test_dataset = CustomDataset(test_data_path, feature_columns, target_column)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader