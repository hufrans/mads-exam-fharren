import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score, precision_score
import pandas as pd
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple
import uuid
import toml

class Trainer:
    """
    A general training framework for PyTorch models.
    Handles training, evaluation, logging, and model saving.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer with configuration from a TOML file.

        Args:
            config_path (str): Path to the TOML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        try:
            self.settings = toml.load(config_path)
        except Exception as e:
            raise ValueError(f"Error loading TOML config: {e}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Construct a unique log directory for this run
        log_base_dir = self.settings["training"]["log_dir"]
        
        # Determine the project root relative to this script for consistent logging
        # This part assumes training_framework.py is in src/fh/
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_script_dir, "..", "..")
        
        # Ensure log_base_dir is relative to the project root
        absolute_log_base_dir = os.path.join(project_root, log_base_dir)

        self.log_dir = os.path.join(absolute_log_base_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(uuid.uuid4())[:8])
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging to: {self.log_dir}")

        self.results_df = pd.DataFrame()

    def _get_optimizer(self, model: nn.Module, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
        """
        Returns the specified optimizer.

        Args:
            model (nn.Module): The model to optimize.
            optimizer_name (str): Name of the optimizer (e.g., "Adam", "SGD").
            learning_rate (float): The initial learning rate.

        Returns:
            optim.Optimizer: The PyTorch optimizer.
        """
        if optimizer_name == "Adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            return optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self, optimizer: optim.Optimizer, scheduler_kwargs: Dict[str, Any]) -> ReduceLROnPlateau:
        """
        Returns a ReduceLROnPlateau learning rate scheduler.

        Args:
            optimizer (optim.Optimizer): The optimizer to schedule.
            scheduler_kwargs (Dict[str, Any]): Keyword arguments for the scheduler,
                                                e.g., 'mode', 'factor', 'patience'.

        Returns:
            ReduceLROnPlateau: The learning rate scheduler.
        """
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("Optimizer must be a PyTorch optimizer.")
        if not isinstance(scheduler_kwargs, dict):
            raise TypeError("Scheduler_kwargs must be a dictionary.")

        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_kwargs.get("mode", "min"),  # Nu haalt deze 'mode' uit kwargs of gebruikt 'min' als default
            factor=scheduler_kwargs.get("factor", 0.1),
            patience=scheduler_kwargs.get("patience", 10)
            # 'verbose=True' is hier verwijderd
        )

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculates recall, accuracy, and precision.

        Args:
            predictions (torch.Tensor): Predicted class labels.
            targets (torch.Tensor): True class labels.

        Returns:
            Dict[str, float]: A dictionary containing calculated metrics.
        """
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Handle multi-class for precision and recall with average='weighted'
        # Check if targets or predictions contain more than 2 unique classes for 'binary' average
        unique_targets = len(set(targets_np))
        if unique_targets > 2:
            average_mode = 'weighted'
        elif unique_targets == 2:
            # For binary classification, 'binary' is often used, but 'weighted' also works.
            # If you want specific positive class metrics, specify `pos_label`.
            # For general purpose, 'weighted' or 'macro' are good defaults for imbalanced datasets.
            average_mode = 'weighted'
        else: # Only one class present (e.g., during testing with a single class batch)
            average_mode = None # Cannot calculate with one class, will result in NaN/error if not handled

        recall = recall_score(targets_np, predictions_np, average=average_mode, zero_division=0)
        accuracy = accuracy_score(targets_np, predictions_np)
        precision = precision_score(targets_np, predictions_np, average=average_mode, zero_division=0)

        return {"recall": recall, "accuracy": accuracy, "precision": precision}

    def train_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """
        Performs one training epoch.

        Args:
            model (nn.Module): The PyTorch model.
            data_loader (DataLoader): DataLoader for training data.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.

        Returns:
            float: The average training loss for the epoch.
        """
        model.train()
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(data_loader.dataset)

    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float, float]:
        """
        Evaluates the model on the given data.

        Args:
            model (nn.Module): The PyTorch model.
            data_loader (DataLoader): DataLoader for evaluation data.
            criterion (nn.Module): The loss function.

        Returns:
            Tuple[float, float, float, float]: A tuple containing average loss, accuracy, recall, and precision.
        """
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        
        # Convert lists to tensors for metric calculation (can be empty if no data in loader)
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)

        if len(predictions_tensor) == 0:
            return avg_loss, 0.0, 0.0, 0.0 # Return zeros if no samples were processed

        metrics = self._calculate_metrics(predictions_tensor, targets_tensor)
        return avg_loss, metrics["accuracy"], metrics["recall"], metrics["precision"]


    def run_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str,
        hyperparams: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs the full training process for a given model.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for test data.
            model_name (str): The name of the model (e.g., "Baseline", "CNN", "GRU").
            hyperparams (Dict[str, Any]): A dictionary of hyperparameters specific to the model and run.

        Returns:
            Dict[str, Any]: A dictionary containing the final training results and metadata.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module.")
        if not isinstance(train_loader, DataLoader) or not isinstance(test_loader, DataLoader):
            raise TypeError("train_loader and test_loader must be PyTorch DataLoaders.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dictionary.")

        start_time = datetime.now()
        model.to(self.device)

        criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
        optimizer = self._get_optimizer(model, self.settings["training"]["optimizer"], self.settings["training"]["learning_rate"])
        
        # De scheduler_kwargs worden nu correct doorgegeven en gebruikt
        scheduler_kwargs = self.settings["training"]["scheduler_kwargs"]
        scheduler = self._get_scheduler(optimizer, scheduler_kwargs)

        epochs = self.settings["training"]["epochs"]
        best_test_loss = float('inf')
        best_model_path = os.path.join(self.log_dir, f"{model_name}_best_model.pth")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            test_loss, accuracy_value, recall_value, precision_value = self.evaluate_epoch(model, test_loader, criterion)
            epoch_duration = time.time() - epoch_start_time

            scheduler.step(test_loss) # Update learning rate based on test loss

            loss_verschil = train_loss - test_loss
            relatief_verschil_loss = (loss_verschil / train_loss) if train_loss != 0 else float('inf')

            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Accuracy: {accuracy_value:.4f}, Recall: {recall_value:.4f}, Precision: {precision_value:.4f}, "
                  f"Duration: {epoch_duration:.2f}s")

            # Prepare data for the dataframe
            epoch_data = {
                'id': str(self.log_dir),
                'model_name': model_name,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'verschil_loss': loss_verschil,
                'relatief_verschil_loss': relatief_verschil_loss,
                'accuracy': accuracy_value,
                'recall': recall_value,
                'precision': precision_value,
                'duration_epoch_seconds': epoch_duration,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **hyperparams # Add model-specific hyperparameters
            }
            self.results_df = pd.concat([self.results_df, pd.DataFrame([epoch_data])], ignore_index=True)

            # Save dataframe for each epoch
            df_output_path = os.path.join(self.log_dir, f"{model_name}_results_epoch_{epoch}.parquet")
            self.results_df.to_parquet(df_output_path, index=False)
            print(f"Saved results for epoch {epoch} to {df_output_path}")

            # Save the best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path} (Test Loss: {best_test_loss:.4f})")

        end_time = datetime.now()
        duration = end_time - start_time

        # Final results dictionary
        final_results = {
            "id": str(self.log_dir),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(duration),
            "model": model_name,
            "epochs": epochs,
            "factor": self.settings["training"]["scheduler_kwargs"].get("factor"),
            "patience": self.settings["training"]["scheduler_kwargs"].get("patience"),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "verschil_loss": loss_verschil,
            "relatief_verschil_loss": relatief_verschil_loss,
            "accuracy": accuracy_value,
            "recall": recall_value,
            "precision": precision_value, # Ensure precision is included in final results
            **hyperparams
        }
        return final_results