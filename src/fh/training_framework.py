import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score, precision_score
import pandas as pd
import os
import time
from datetime import datetime, timedelta # Importeer timedelta
from typing import Dict, Any, Tuple
import uuid
import toml
import shutil
from loguru import logger # Importeer loguru's logger
import sys # Nodig voor logger.add(sys.stderr, ...)

class Trainer:
    """
    A general training framework for PyTorch models.

    This class provides functionalities for training, evaluating, and logging
    machine learning models built with PyTorch. It handles optimizer and
    scheduler setup, calculates various metrics, saves the best performing
    model, and logs training results incrementally to disk with a rotating backup.
    Logging is managed using Loguru, ensuring detailed and organized output.

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
        It also configures Loguru to log to a specific file within the run's
        log directory, and to the console.

        Args:
            config_path (str): Path to the TOML configuration file.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            ValueError: If there is an error loading or parsing the TOML configuration.
        """
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at: {config_path}")
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        try:
            self.settings: Dict[str, Any] = toml.load(config_path)
        except Exception as e:
            logger.error(f"Error loading TOML config from {config_path}: {e}")
            raise ValueError(f"Error loading TOML config: {e}")

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Construct a unique log directory for this run
        log_base_dir: str = self.settings["training"]["log_dir"]
        
        # Determine the project root relative to this script for consistent logging
        current_script_dir: str = os.path.dirname(os.path.abspath(__file__))
        project_root: str = os.path.join(current_script_dir, "..", "..")
        
        # Ensure log_base_dir is relative to the project root
        absolute_log_base_dir: str = os.path.join(project_root, log_base_dir)

        self.log_dir: str = os.path.join(absolute_log_base_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(uuid.uuid4())[:8])
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Logging for this run will be stored in: {self.log_dir}")

        # Configure Loguru for this specific Trainer instance:
        # Remove default handler to avoid duplicate logs if Trainer is instantiated multiple times
        # This is crucial for isolated logging per Trainer instance.
        logger.remove() 
        # Add handler for file logging within the unique log_dir
        logger.add(os.path.join(self.log_dir, "training_log_{time}.log"), 
                   level="INFO", 
                   rotation="10 MB", # Rotate log file if it exceeds 10 MB
                   retention="7 days", # Keep logs for 7 days
                   compression="zip", # Compress old log files
                   serialize=False) # Keep logs human-readable
        # Add handler for console output (INFO level and above)
        logger.add(sys.stderr, level="INFO") # Output INFO level and above to console


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
        logger.debug(f"Attempting to get optimizer: {optimizer_name} with learning rate: {learning_rate}")
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            logger.debug("Adam optimizer created.")
            return optimizer
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            logger.debug("SGD optimizer created.")
            return optimizer
        else:
            logger.error(f"Unsupported optimizer specified: {optimizer_name}")
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
        logger.debug(f"Attempting to get scheduler with kwargs: {scheduler_kwargs}")
        if not isinstance(optimizer, optim.Optimizer):
            logger.error(f"Invalid optimizer type provided: {type(optimizer)}. Expected torch.optim.Optimizer.")
            raise TypeError("Optimizer must be a PyTorch optimizer.")
        if not isinstance(scheduler_kwargs, dict):
            logger.error(f"Invalid scheduler_kwargs type provided: {type(scheduler_kwargs)}. Expected dictionary.")
            raise TypeError("Scheduler_kwargs must be a dictionary.")

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_kwargs.get("mode", "min"),
            factor=scheduler_kwargs.get("factor", 0.1),
            patience=scheduler_kwargs.get("patience", 10)
        )
        logger.debug(f"ReduceLROnPlateau scheduler created with mode='{scheduler.mode}', factor={scheduler.factor}, patience={scheduler.patience}.")
        return scheduler

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

        unique_targets: int = targets_np.nunique()
        if unique_targets > 1:
            average_mode: str = 'weighted'
            recall: float = recall_score(targets_np, predictions_np, average=average_mode, zero_division=0)
            precision: float = precision_score(targets_np, predictions_np, average=average_mode, zero_division=0)
            logger.debug(f"Calculated weighted recall and precision (unique targets > 1).")
        else: # Handle cases with only one unique class in targets
            recall: float = 0.0 
            precision: float = 0.0 
            logger.warning(f"Only 1 unique class ({targets_np.iloc[0]}) in targets. Recall and Precision set to 0.0.")
        
        accuracy: float = accuracy_score(targets_np, predictions_np)
        logger.debug(f"Calculated metrics: Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}.")
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
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            logger.debug(f"Batch {batch_idx+1}/{len(data_loader)}: Loss={loss.item():.4f}")

        avg_loss = total_loss / len(data_loader.dataset)
        logger.info(f"Training Epoch finished. Average Train Loss: {avg_loss:.4f}")
        return avg_loss

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
        model.eval()
        total_loss: float = 0.0
        all_predictions: list[Any] = []
        all_targets: list[Any] = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                logger.debug(f"Evaluation Batch {batch_idx+1}/{len(data_loader)}")

        avg_loss: float = total_loss / len(data_loader.dataset)
        
        predictions_tensor: torch.Tensor = torch.tensor(all_predictions)
        targets_tensor: torch.Tensor = torch.tensor(all_targets)

        if len(predictions_tensor) == 0:
            logger.warning("No predictions made during evaluation. Returning 0 for metrics.")
            return avg_loss, 0.0, 0.0, 0.0

        metrics: Dict[str, float] = self._calculate_metrics(predictions_tensor, targets_tensor)
        logger.info(f"Evaluation Epoch finished. Average Test Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
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
        # --- Input validation and initial setup ---
        logger.info(f"Starting training run for model: {model_name}")
        if not isinstance(model, nn.Module):
            logger.error(f"Invalid model type provided: {type(model)}. Expected torch.nn.Module.")
            raise TypeError("Model must be a PyTorch nn.Module.")
        if not isinstance(train_loader, DataLoader) or not isinstance(test_loader, DataLoader):
            logger.error(f"Invalid DataLoader type provided. train_loader: {type(train_loader)}, test_loader: {type(test_loader)}.")
            raise TypeError("train_loader and test_loader must be PyTorch DataLoaders.")
        if not isinstance(model_name, str) or not model_name:
            logger.error(f"Invalid model_name: '{model_name}'. Must be a non-empty string.")
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(hyperparams, dict):
            logger.error(f"Invalid hyperparams type provided: {type(hyperparams)}. Expected dictionary.")
            raise TypeError("hyperparams must be a dictionary.")

        start_time: datetime = datetime.now()
        model.to(self.device)
        logger.info(f"Model {model_name} moved to device: {self.device}.")
        logger.debug(f"Hyperparameters for {model_name}: {hyperparams}")

        criterion: nn.Module = nn.CrossEntropyLoss() 
        optimizer: optim.Optimizer = self._get_optimizer(model, self.settings["training"]["optimizer"], self.settings["training"]["learning_rate"])
        
        scheduler_kwargs: Dict[str, Any] = self.settings["training"]["scheduler_kwargs"]
        scheduler: ReduceLROnPlateau = self._get_scheduler(optimizer, scheduler_kwargs)

        epochs: int = self.settings["training"]["epochs"]
        best_test_loss: float = float('inf')
        best_model_path: str = os.path.join(self.log_dir, f"{model_name}_best_model.pth")
        
        best_epoch_metrics: Dict[str, Any] = {} 

        model_results_parquet_path: str = os.path.join(self.log_dir, f"{model_name}_results.parquet")
        model_results_backup_path: str = f"{model_results_parquet_path}.bak"
        logger.debug(f"Results will be saved to: {model_results_parquet_path}")

        # --- Training Loop ---
        for epoch in range(1, epochs + 1):
            epoch_start_time: float = time.time()
            logger.info(f"--- Epoch {epoch}/{epochs} for {model_name} ---")
            
            # Train and evaluate
            train_loss: float = self.train_epoch(model, train_loader, criterion, optimizer)
            test_loss, accuracy_value, recall_value, precision_value = self.evaluate_epoch(model, test_loader, criterion)
            epoch_duration: float = time.time() - epoch_start_time

            scheduler.step(test_loss) # Update learning rate

            loss_verschil: float = train_loss - test_loss
            relatief_verschil_loss: float = (loss_verschil / train_loss) if train_loss != 0 else float('inf')

            logger.info(f"Epoch {epoch} Summary for {model_name}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                        f"Accuracy: {accuracy_value:.4f}, Recall: {recall_value:.4f}, Precision: {precision_value:.4f}, "
                        f"Duration: {epoch_duration:.2f}s, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Prepare data for the dataframe
            epoch_data: Dict[str, Any] = {
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
                **hyperparams 
            }
            
            # --- Rotating Backup Mechanism ---
            if os.path.exists(model_results_parquet_path):
                try:
                    shutil.copy(model_results_parquet_path, model_results_backup_path)
                    logger.debug(f"Backed up current results from {model_results_parquet_path} to {model_results_backup_path}.")
                except Exception as e:
                    logger.warning(f"Failed to create backup for {model_results_parquet_path}. Error: {e}")
            
            # --- Read, Update, and Write Results to Disk ---
            current_df: pd.DataFrame
            if os.path.exists(model_results_parquet_path):
                try:
                    existing_df: pd.DataFrame = pd.read_parquet(model_results_parquet_path)
                    current_df = pd.concat([existing_df, pd.DataFrame([epoch_data])], ignore_index=True)
                    logger.debug(f"Appended epoch {epoch} data to existing DataFrame for {model_name}.")
                except Exception as e:
                    logger.error(f"Failed to read existing Parquet file {model_results_parquet_path}. Starting new DataFrame. Error: {e}")
                    current_df = pd.DataFrame([epoch_data])
            else:
                current_df = pd.DataFrame([epoch_data])
                logger.debug(f"Created new DataFrame for epoch {epoch} results for {model_name}.")
            
            try:
                current_df.to_parquet(model_results_parquet_path, index=False)
                logger.info(f"Results for epoch {epoch} saved to {model_results_parquet_path}.")
            except Exception as e:
                logger.error(f"Failed to save results for epoch {epoch} to {model_results_parquet_path}. Error: {e}")


            # Save the best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.success(f"Saved best model for {model_name} to {best_model_path} (Test Loss: {best_test_loss:.4f}).")
                    best_epoch_metrics = { 
                        "best_test_loss": test_loss,
                        "best_accuracy": accuracy_value,
                        "best_recall": recall_value,
                        "best_precision": precision_value,
                        "best_epoch": epoch
                    }
                except Exception as e:
                    logger.error(f"Failed to save best model for {model_name} to {best_model_path}. Error: {e}")
            else:
                logger.debug(f"Current test loss {test_loss:.4f} not better than best {best_test_loss:.4f}. Model not saved.")
            
            # Optional: Implement early stopping based on self.settings
            # if early_stopping_condition_met:
            #     logger.info(f"Early stopping triggered at epoch {epoch} for model {model_name}.")
            #     break

        end_time: datetime = datetime.now()
        duration: timedelta = end_time - start_time
        logger.info(f"Training for model {model_name} finished. Total duration: {duration}")

        # Final results dictionary
        final_results: Dict[str, Any] = {
            "id": str(self.log_dir),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(duration),
            "model": model_name,
            "epochs_run": epoch, # The actual number of epochs run (could be less due to early stopping)
            "factor": self.settings["training"]["scheduler_kwargs"].get("factor"),
            "patience": self.settings["training"]["scheduler_kwargs"].get("patience"),
            **best_epoch_metrics, 
            **hyperparams 
        }
        logger.debug(f"Final summary results for {model_name}: {final_results}")
        return final_results