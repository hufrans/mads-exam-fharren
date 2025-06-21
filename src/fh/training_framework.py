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
import shutil

class Trainer:
    """
    A general training framework for PyTorch models.

    This class provides functionalities for training, evaluating, and logging
    machine learning models built with PyTorch. It handles optimizer and
    scheduler setup, calculates various metrics, saves the best performing
    model, and logs training results incrementally to disk with a rotating backup.

    Attributes:
        settings (Dict[str, Any]): A dictionary containing all configuration settings
                                   loaded from the TOML file.
        device (torch.device): The device (CPU or CUDA) on which the model will be trained.
        log_dir (str): The unique directory path where all logs, model checkpoints,
                       and results will be stored for the current training run.
        results_df (pd.DataFrame): An initialized (empty) DataFrame. While not used for
                                   in-memory aggregation anymore, it serves as a placeholder
                                   or for potential future aggregate logging at the end of a run.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer with configuration from a TOML file.

        The constructor sets up the device, creates a unique logging directory,
        and loads all training parameters from the specified TOML configuration.

        Args:
            config_path (str): Path to the TOML configuration file.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            ValueError: If there is an error loading or parsing the TOML configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        try:
            self.settings: Dict[str, Any] = toml.load(config_path)
        except Exception as e:
            raise ValueError(f"Error loading TOML config: {e}")

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Construct a unique log directory for this run
        log_base_dir: str = self.settings["training"]["log_dir"]
        
        # Determine the project root relative to this script for consistent logging
        current_script_dir: str = os.path.dirname(os.path.abspath(__file__))
        project_root: str = os.path.join(current_script_dir, "..", "..")
        
        # Ensure log_base_dir is relative to the project root
        absolute_log_base_dir: str = os.path.join(project_root, log_base_dir)

        self.log_dir: str = os.path.join(absolute_log_base_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(uuid.uuid4())[:8])
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging to: {self.log_dir}")

        self.results_df: pd.DataFrame = pd.DataFrame() 

    def _get_optimizer(self, model: nn.Module, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
        """
        Returns the specified PyTorch optimizer for the given model.

        Args:
            model (nn.Module): The PyTorch model whose parameters are to be optimized.
            optimizer_name (str): The name of the optimizer (e.g., "Adam", "SGD").
            learning_rate (float): The initial learning rate for the optimizer.

        Returns:
            optim.Optimizer: An initialized PyTorch optimizer object.

        Raises:
            ValueError: If an unsupported optimizer name is provided.
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

        This scheduler reduces the learning rate when a metric (e.g., validation loss)
        has stopped improving.

        Args:
            optimizer (optim.Optimizer): The optimizer to which the scheduler will be linked.
            scheduler_kwargs (Dict[str, Any]): A dictionary of keyword arguments for the
                                                ReduceLROnPlateau scheduler. Expected keys
                                                include 'mode' (e.g., 'min'), 'factor', and 'patience'.

        Returns:
            ReduceLROnPlateau: An initialized ReduceLROnPlateau scheduler object.

        Raises:
            TypeError: If optimizer is not a PyTorch optimizer or scheduler_kwargs is not a dictionary.
        """
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("Optimizer must be a PyTorch optimizer.")
        if not isinstance(scheduler_kwargs, dict):
            raise TypeError("Scheduler_kwargs must be a dictionary.")

        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_kwargs.get("mode", "min"),
            factor=scheduler_kwargs.get("factor", 0.1),
            patience=scheduler_kwargs.get("patience", 10)
        )

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculates recall, accuracy, and precision for classification tasks.

        For multi-class classification, recall and precision are calculated with
        'weighted' averaging. If only one unique class is present in targets,
        recall and precision will be 0.0 to avoid errors.

        Args:
            predictions (torch.Tensor): A 1D tensor of predicted class labels.
            targets (torch.Tensor): A 1D tensor of true class labels.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics:
                              'recall', 'accuracy', and 'precision'.
        """
        predictions_np: pd.Series = pd.Series(predictions.cpu().numpy())
        targets_np: pd.Series = pd.Series(targets.cpu().numpy())

        # Ensure that there are multiple unique classes for weighted average calculation
        # If not, metrics like recall/precision might raise errors or be meaningless.
        unique_targets: int = targets_np.nunique()
        if unique_targets > 1:
            average_mode: str = 'weighted'
            recall: float = recall_score(targets_np, predictions_np, average=average_mode, zero_division=0)
            precision: float = precision_score(targets_np, predictions_np, average=average_mode, zero_division=0)
        else: # Handle cases with only one unique class in targets
            recall: float = 0.0 
            precision: float = 0.0 
        
        accuracy: float = accuracy_score(targets_np, predictions_np)

        return {"recall": recall, "accuracy": accuracy, "precision": precision}

    def train_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """
        Performs one full training epoch over the provided dataset.

        Sets the model to training mode, iterates through the data_loader,
        performs forward and backward passes, and updates model weights.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            data_loader (DataLoader): DataLoader containing the training data.
            criterion (nn.Module): The loss function used for training.
            optimizer (optim.Optimizer): The optimizer used to update model weights.

        Returns:
            float: The average training loss across all batches for the epoch.
        """
        model.train()
        total_loss: float = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update model parameters
            total_loss += loss.item() * inputs.size(0) # Accumulate loss

        return total_loss / len(data_loader.dataset) # Return average loss

    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float, float]:
        """
        Evaluates the model's performance on a given dataset for one epoch.

        Sets the model to evaluation mode, iterates through the data_loader
        without computing gradients, and calculates loss and key metrics.

        Args:
            model (nn.Module): The PyTorch model to be evaluated.
            data_loader (DataLoader): DataLoader containing the evaluation data.
            criterion (nn.Module): The loss function used for evaluation.

        Returns:
            Tuple[float, float, float, float]: A tuple containing:
                - avg_loss (float): The average evaluation loss for the epoch.
                - accuracy (float): The accuracy score.
                - recall (float): The recall score.
                - precision (float): The precision score.
        """
        model.eval() # Set model to evaluation mode
        total_loss: float = 0.0
        all_predictions: list[Any] = []
        all_targets: list[Any] = []

        with torch.no_grad(): # Disable gradient computation for evaluation
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1) # Get predicted class
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss: float = total_loss / len(data_loader.dataset)
        
        # Convert lists to tensors for metric calculation
        predictions_tensor: torch.Tensor = torch.tensor(all_predictions)
        targets_tensor: torch.Tensor = torch.tensor(all_targets)

        # Handle case where no predictions were made (e.g., empty data_loader)
        if len(predictions_tensor) == 0:
            return avg_loss, 0.0, 0.0, 0.0

        metrics: Dict[str, float] = self._calculate_metrics(predictions_tensor, targets_tensor)
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
        Orchestrates the full training process for a given model.

        This method iterates through epochs, calling train_epoch and evaluate_epoch,
        applies learning rate scheduling, saves the best model checkpoint, and
        logs detailed epoch-wise results directly to a Parquet file on disk.
        A single rotating backup of the results file is maintained.

        Args:
            model (nn.Module): The PyTorch model instance to train.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test/validation dataset.
            model_name (str): A unique name for the model being trained (e.g., "Baseline", "CNN_Exp1").
                              This name is used for logging and file naming.
            hyperparams (Dict[str, Any]): A dictionary of model-specific hyperparameters
                                          and any other relevant parameters to be logged
                                          with each epoch's results.

        Returns:
            Dict[str, Any]: A summary dictionary of the best performing epoch's metrics
                            and overall training run metadata.

        Raises:
            TypeError: If model, train_loader, test_loader, or hyperparams are of incorrect types.
            ValueError: If model_name is not a non-empty string.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module.")
        if not isinstance(train_loader, DataLoader) or not isinstance(test_loader, DataLoader):
            raise TypeError("train_loader and test_loader must be PyTorch DataLoaders.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(hyperparams, dict):
            raise TypeError("hyperparams must be a dictionary.")

        start_time: datetime = datetime.now()
        model.to(self.device)

        criterion: nn.Module = nn.CrossEntropyLoss() # Standard loss function for multi-class classification
        optimizer: optim.Optimizer = self._get_optimizer(model, self.settings["training"]["optimizer"], self.settings["training"]["learning_rate"])
        
        scheduler_kwargs: Dict[str, Any] = self.settings["training"]["scheduler_kwargs"]
        scheduler: ReduceLROnPlateau = self._get_scheduler(optimizer, scheduler_kwargs)

        epochs: int = self.settings["training"]["epochs"]
        best_test_loss: float = float('inf')
        best_model_path: str = os.path.join(self.log_dir, f"{model_name}_best_model.pth")
        
        best_epoch_metrics: Dict[str, Any] = {} # To store metrics of the best epoch

        # Define the path for the single Parquet results file for THIS model training run
        model_results_parquet_path: str = os.path.join(self.log_dir, f"{model_name}_results.parquet")
        # Define the path for the single rotating backup file
        model_results_backup_path: str = f"{model_results_parquet_path}.bak"


        for epoch in range(1, epochs + 1):
            epoch_start_time: float = time.time()
            train_loss: float = self.train_epoch(model, train_loader, criterion, optimizer)
            test_loss, accuracy_value, recall_value, precision_value = self.evaluate_epoch(model, test_loader, criterion)
            epoch_duration: float = time.time() - epoch_start_time

            scheduler.step(test_loss) # Update learning rate based on test loss

            loss_verschil: float = train_loss - test_loss
            # Calculate relative difference, handle division by zero if train_loss is 0
            relatief_verschil_loss: float = (loss_verschil / train_loss) if train_loss != 0 else float('inf')

            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Accuracy: {accuracy_value:.4f}, Recall: {recall_value:.4f}, Precision: {precision_value:.4f}, "
                  f"Duration: {epoch_duration:.2f}s")

            # Prepare data for the dataframe
            epoch_data: Dict[str, Any] = {
                'id': str(self.log_dir), # Unique ID for the entire run
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
                **hyperparams # Add model-specific hyperparameters EXACTLY as provided
            }
            
            # --- Correct Rotating Backup Mechanism ---
            # If the main results file exists, copy its CURRENT state to the single backup file.
            # This overwrites any previous backup, ensuring 'model_results_backup_path'
            # always holds the state just before the latest update.
            if os.path.exists(model_results_parquet_path):
                try:
                    shutil.copy(model_results_parquet_path, model_results_backup_path)
                    print(f"Backed up {model_results_parquet_path} to {model_results_backup_path} (overwriting previous backup).")
                except Exception as e:
                    print(f"Warning: Could not create backup for {model_results_parquet_path}. Error: {e}")
            # --- End Rotating Backup Mechanism ---

            # Read the existing (now potentially backed up) Parquet file,
            # append the new row, and write it back.
            current_df: pd.DataFrame
            if os.path.exists(model_results_parquet_path):
                existing_df: pd.DataFrame = pd.read_parquet(model_results_parquet_path)
                current_df = pd.concat([existing_df, pd.DataFrame([epoch_data])], ignore_index=True)
            else:
                # If the file does not exist (first epoch), start with a new DataFrame
                current_df = pd.DataFrame([epoch_data])
            
            # Write the updated DataFrame to disk, overwriting the original file
            current_df.to_parquet(model_results_parquet_path, index=False)
            print(f"Results for epoch {epoch} appended to and saved to {model_results_parquet_path}")


            # Save the best model based on test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path} (Test Loss: {best_test_loss:.4f})")
                # Store metrics of this best epoch
                best_epoch_metrics = { 
                    "best_test_loss": test_loss,
                    "best_accuracy": accuracy_value,
                    "best_recall": recall_value,
                    "best_precision": precision_value,
                    "best_epoch": epoch
                }
            # Early stopping check can be added here if desired (based on self.settings)

        end_time: datetime = datetime.now()
        duration: timedelta = end_time - start_time # Corrected type to timedelta

        # Final results dictionary - Use the metrics from the best epoch
        final_results: Dict[str, Any] = {
            "id": str(self.log_dir),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(duration),
            "model": model_name,
            "epochs_run": epoch, # The actual number of epochs run (could be less due to early stopping)
            "factor": self.settings["training"]["scheduler_kwargs"].get("factor"),
            "patience": self.settings["training"]["scheduler_kwargs"].get("patience"),
            **best_epoch_metrics, # Add the metrics from the best epoch
            **hyperparams # INCLUDE ALL PROVIDED HYPERPARAMS EXACTLY
        }
        return final_results