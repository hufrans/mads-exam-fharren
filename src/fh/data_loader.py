import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from loguru import logger

class ParquetDataset(Dataset):
    """
    Een aangepaste PyTorch Dataset voor Parquet DataFrames.
    Deze klasse accepteert een reeds geladen DataFrame, scheidt features en targets,
    en past optioneel standaardisatie toe.
    """
    def __init__(self, dataframe: pd.DataFrame, feature_columns: List[str], target_column: str, scaler: StandardScaler = None):
        """
        Initialiseert de dataset.

        Args:
            dataframe (pd.DataFrame): De reeds geladen DataFrame.
            feature_columns (List[str]): Een lijst van kolomnamen die als features moeten worden gebruikt.
            target_column (str): De naam van de kolom die als target moet worden gebruikt.
            scaler (StandardScaler, optioneel): Een getrainde StandardScaler om features te transformeren.
                                                Indien None, vindt er geen schaling plaats.
        """
        self.df = dataframe
        logger.debug(f"Dataset geïnitialiseerd met DataFrame. Aantal rijen: {len(self.df)}")

        # Controleer of alle feature_columns en target_column aanwezig zijn in de DataFrame
        missing_features = [col for col in feature_columns if col not in self.df.columns]
        if missing_features:
            logger.error(f"Ontbrekende feature kolommen in data: {missing_features}")
            raise ValueError(f"Missing feature columns in data: {missing_features}")
        
        if target_column not in self.df.columns:
            logger.error(f"Ontbrekende target kolom '{target_column}' in data.")
            raise ValueError(f"Missing target column '{target_column}' in data.")

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler = scaler

        self.features = self.df[self.feature_columns].values
        self.targets = self.df[self.target_column].values

        if self.scaler is not None:
            logger.debug("Scaler wordt toegepast op features.")
            self.features = self.scaler.transform(self.features)
        
        # Converteer naar PyTorch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long) # Targets zijn doorgaans long voor classificatie

    def __len__(self) -> int:
        """Retourneert het totale aantal samples in de dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Haalt een sample en het bijbehorende target op basis van de index.

        Args:
            idx (int): De index van het sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Een tuple met de feature-tensor en de target-tensor.
        """
        return self.features[idx], self.targets[idx]

def get_data_loaders(
    train_data_path: str, # Nu weer een enkel pad
    test_data_path: str,
    feature_columns: List[str],
    target_column: str,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]: # Retourneert ook de DataFrames
    """
    Laadt trainings- en testdata en creëert PyTorch DataLoaders.

    Args:
        train_data_path (str): Pad naar het trainings Parquet-bestand.
        test_data_path (str): Pad naar het test Parquet-bestand.
        feature_columns (List[str]): Kolomnamen voor features.
        target_column (str): Kolomnaam voor targets.
        batch_size (int): Batchgrootte voor DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
            - train_loader (DataLoader): DataLoader voor trainingsdata.
            - test_loader (DataLoader): DataLoader voor testdata.
            - train_df (pd.DataFrame): De trainings DataFrame.
            - test_df (pd.DataFrame): De test DataFrame.
    """
    logger.info(f"Start aanmaken data loaders. Training bestand: {train_data_path}, Test bestand: {test_data_path}")

    # Laad trainingsdata (geen concatenatie meer hier)
    try:
        train_df = pd.read_parquet(train_data_path)
        logger.info(f"Succesvol trainingsbestand '{train_data_path}' geladen ({len(train_df)} rijen).")
    except FileNotFoundError:
        logger.error(f"Trainingsbestand niet gevonden: {train_data_path}. Zorg ervoor dat het pad correct is.")
        raise FileNotFoundError(f"Trainingsbestand niet gevonden: {train_data_path}")
    except Exception as e:
        logger.error(f"Fout bij het laden van trainingsbestand {train_data_path}: {e}")
        raise Exception(f"Fout bij het laden van trainingsbestand {train_data_path}: {e}")

    # Laad testdata
    try:
        test_df = pd.read_parquet(test_data_path)
        logger.info(f"Succesvol testbestand '{test_data_path}' geladen ({len(test_df)} rijen).")
    except FileNotFoundError:
        logger.error(f"Testbestand niet gevonden: {test_data_path}. Zorg ervoor dat het pad correct is.")
        raise FileNotFoundError(f"Testbestand niet gevonden: {test_data_path}")
    except Exception as e:
        logger.error(f"Fout bij het laden van testbestand {test_data_path}: {e}")
        raise Exception(f"Fout bij het laden van testbestand {test_data_path}: {e}")

    # Initialiseer en train StandardScaler op de trainingsfeatures
    scaler = StandardScaler()
    try:
        # Controleer of alle feature_columns aanwezig zijn in de trainings DataFrame
        missing_features_train = [col for col in feature_columns if col not in train_df.columns]
        if missing_features_train:
            logger.error(f"Ontbrekende feature kolommen in training data: {missing_features_train}")
            raise ValueError(f"Missing feature columns in training data: {missing_features_train}")
        
        # Controleer of target_column aanwezig is in de trainings DataFrame
        if target_column not in train_df.columns:
            logger.error(f"Ontbrekende target kolom '{target_column}' in training data.")
            raise ValueError(f"Missing target column '{target_column}' in training data.")

        scaler.fit(train_df[feature_columns].values)
        logger.info("StandardScaler getraind op trainingsdata.")
    except Exception as e:
        logger.error(f"Fout bij het fitten van StandardScaler op trainingsdata: {e}")
        raise Exception(f"Fout bij het fitten van StandardScaler: {e}")

    # Creëer datasets met de getrainde scaler
    try:
        train_dataset = ParquetDataset(train_df, feature_columns, target_column, scaler)
        test_dataset = ParquetDataset(test_df, feature_columns, target_column, scaler)
        logger.info("Datasets succesvol aangemaakt.")
    except Exception as e:
        logger.error(f"Fout bij het aanmaken van datasets: {e}")
        raise Exception(f"Fout bij het aanmaken van datasets: {e}")

    # Creëer DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("DataLoaders succesvol aangemaakt.")

    return train_loader, test_loader, train_df, test_df
