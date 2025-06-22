import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
from loguru import logger # Importeer loguru's logger

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
        logger.info(f"Initializing CustomDataset from: {data_path}")
        try:
            # Lees Parquet-bestand in plaats van CSV
            self.data = pd.read_parquet(data_path)
            logger.success(f"Data successfully loaded from {data_path}. Found {len(self.data)} samples.")
        except FileNotFoundError:
            logger.error(f"Data file not found at: {data_path}")
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise IOError(f"Error loading data from {data_path}: {e}")

        if not all(col in self.data.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in self.data.columns]
            logger.error(f"Missing feature columns in data from {data_path}: {missing_cols}")
            raise ValueError(f"Missing feature columns in data: {missing_cols}")
        if target_column not in self.data.columns:
            logger.error(f"Missing target column '{target_column}' in data from {data_path}.")
            raise ValueError(f"Missing target column '{target_column}' in data.")

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values
        logger.debug(f"Features shape: {self.features.shape}, Targets shape: {self.targets.shape}")

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
        # logger.debug(f"Retrieving item {idx}: features shape {features.shape}, target {targets}") # Te veel logs voor elke item
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
    logger.info(f"Creating data loaders with batch_size={batch_size}.")
    logger.debug(f"Train data path: {train_data_path}")
    logger.debug(f"Test data path: {test_data_path}")
    logger.debug(f"Feature columns: {feature_columns}")
    logger.debug(f"Target column: {target_column}")

    if not isinstance(batch_size, int) or batch_size <= 0:
        logger.error(f"Invalid batch_size provided: {batch_size}. Must be a positive integer.")
        raise ValueError("batch_size must be a positive integer.")

    try:
        train_dataset = CustomDataset(train_data_path, feature_columns, target_column)
        test_dataset = CustomDataset(test_data_path, feature_columns, target_column)
    except (FileNotFoundError, IOError, ValueError) as e:
        logger.critical(f"Failed to create datasets due to data loading error: {e}")
        raise # Re-raise the exception after logging

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.success(f"Successfully created train loader with {len(train_loader.dataset)} samples "
                   f"and test loader with {len(test_loader.dataset)} samples.")
    return train_loader, test_loader