import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR # Import StepLR if you want to use it
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score # Added f1_score
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import shutil
from loguru import logger
import sys

# Importeer de functie om modellen te selecteren
from src.fh.model_selector import get_model

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
        run_id (str): The unique identifier for the current training run, passed from main.py.
        run_log_dir (str): The unique directory path where all logs, model checkpoints,
                           and results will be stored for the current training run.
        best_model_path_template (str): Template for saving the best model checkpoint.
        results_path_template (str): Template for saving epoch-wise results.
        _log_handler_ids (list): Internal list to keep track of Loguru handler IDs added by this Trainer instance.
        train_data_path (str): Absolute path to the training data file.
        test_data_path (str): Absolute path to the test data file.
        class_weights (torch.Tensor): Tensor of weights for each class, used in the loss function.
    """
    def __init__(self, config: Dict[str, Any], feature_count: int, class_count: int, log_base_dir: str, run_id: str, train_data_path: str, test_data_path: str, class_weights: torch.Tensor) -> None:
        """
        Initializes the Trainer with configuration and data-related parameters.

        The constructor sets up the device, creates a unique logging directory,
        and loads all training parameters from the specified configuration dictionary.
        It also configures Loguru to log to a specific file within the run's
        log directory, and to the console.

        Args:
            config (Dict[str, Any]): Een dictionary met alle configuratieparameters.
            feature_count (int): Het aantal features in de inputdata.
            class_count (int): Het aantal klassen in de output (voor classificatie).
            log_base_dir (str): Het basispad voor alle logbestanden (bijv. 'project_root/logs').
            run_id (str): De unieke ID voor de huidige run, doorgegeven vanuit main.py.
            train_data_path (str): Absolute path to the training data file.
            test_data_path (str): Absolute path to the test data file.
            class_weights (torch.Tensor): Tensor of weights for each class, used in the loss function.
        """
        self.settings = config
        self.feature_count = feature_count
        self.class_count = class_count
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.run_id = run_id
        self.train_data_path = train_data_path # Store the path
        self.test_data_path = test_data_path   # Store the path
        self.class_weights = class_weights.to(self.device) # Store and move class weights to device
        
        # De run-specifieke log directory begint nu met de run_id
        self.run_log_dir = os.path.join(log_base_dir, f"{self.run_id}_run_logs")
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # Bestandsnamen beginnen met run_id
        self.best_model_path_template = os.path.join(self.run_log_dir, f"{self.run_id}_{{model_name}}_best_model.pth")
        self.results_path_template = os.path.join(self.run_log_dir, f"{self.run_id}_{{model_name}}_results.parquet")

        # --- Loguru configuratie voor de Trainer ---
        # We slaan de IDs van de handlers op die we ZELF toevoegen
        self._log_handler_ids = []

        # Voeg een unieke file handler toe voor deze specifieke run (geprefixeerd met run_id)
        # We slaan de ID op, zodat deze later expliciet kan worden verwijderd na de training.
        file_handler_id = logger.add(os.path.join(self.run_log_dir, f"{self.run_id}_trainer_run_training.log"), 
                                     level="DEBUG", # BELANGRIJK: Zet dit op DEBUG voor gedetailleerde logs van de training
                                     rotation="10 MB", 
                                     retention="7 days", 
                                     compression="zip", 
                                     serialize=False,
                                     enqueue=True) # Essentieel voor thread-safety
        self._log_handler_ids.append(file_handler_id)
        
        # OPMERKING: De Trainer is NIET verantwoordelijk voor het toevoegen van de console logger.
        # Dit wordt afgehandeld door main.py om duplicatie te voorkomen en globaal consistent te zijn.
        # De code om te controleren op en potentieel een console handler toe te voegen is hier verwijderd.

        logger.info(f"Initializing Trainer. Using device: {self.device}")
        logger.info(f"Logging and outputs for this run will be stored in: {self.run_log_dir}")
        logger.debug(f"Best model path template: {self.best_model_path_template}")
        logger.debug(f"Results path template: {self.results_path_template}")

    def _get_optimizer(self, model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0) -> optim.Optimizer:
        """
        Returns the specified PyTorch optimizer for the given model.

        Args:
            model (nn.Module): The PyTorch model whose parameters are to be optimized.
            optimizer_name (str): The name of the optimizer (e.g., "Adam", "SGD").
            learning_rate (float): The initial learning rate for the optimizer.
            weight_decay (float): L2 regularization factor.

        Returns:
            optim.Optimizer: An initialized PyTorch optimizer object.

        Raises:
            ValueError: If an unsupported optimizer name is provided.
        """
        logger.debug(f"Attempting to get optimizer: {optimizer_name} with learning rate: {learning_rate}, weight_decay: {weight_decay}")
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            logger.debug("Adam optimizer created.")
            return optimizer
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            logger.debug("SGD optimizer created.")
            return optimizer
        else:
            logger.error(f"Unsupported optimizer specified: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self, optimizer: optim.Optimizer, scheduler_type: str, scheduler_kwargs: Dict[str, Any]) -> Any:
        """
        Returns a learning rate scheduler based on the specified type.

        Args:
            optimizer (optim.Optimizer): The optimizer to which the scheduler will be linked.
            scheduler_type (str): The type of scheduler (e.g., "ReduceLROnPlateau", "StepLR").
            scheduler_kwargs (Dict[str, Any]): A dictionary of keyword arguments for the scheduler.

        Returns:
            Any: An initialized PyTorch scheduler object or None if no valid scheduler type.

        Raises:
            TypeError: If optimizer is not a PyTorch optimizer or scheduler_kwargs is not a dictionary.
        """
        logger.debug(f"Attempting to get scheduler: {scheduler_type} with kwargs: {scheduler_kwargs}")
        if not isinstance(optimizer, optim.Optimizer):
            logger.error(f"Invalid optimizer type provided: {type(optimizer)}. Expected torch.optim.Optimizer.")
            raise TypeError("Optimizer must be a PyTorch optimizer.")
        if not isinstance(scheduler_kwargs, dict):
            logger.error(f"Invalid scheduler_kwargs type provided: {type(scheduler_kwargs)}. Expected dictionary.")
            raise TypeError("Scheduler_kwargs must be a dictionary.")

        scheduler = None
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_kwargs.get("mode", "min"),
                factor=scheduler_kwargs.get("factor", 0.1),
                patience=scheduler_kwargs.get("patience", 10),
                verbose=True # Voeg verbose toe voor logmeldingen van scheduler
            )
            logger.debug(f"ReduceLROnPlateau scheduler created with mode='{scheduler.mode}', factor={scheduler.factor}, patience={scheduler.patience}.")
        elif scheduler_type == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_kwargs.get("step_size", 10),
                gamma=scheduler_kwargs.get("gamma", 0.1)
            )
            logger.debug(f"StepLR scheduler created with step_size={scheduler.step_size}, gamma={scheduler.gamma}.")
        else:
            logger.warning(f"Unsupported scheduler type specified: {scheduler_type}. No scheduler will be used.")
        
        return scheduler

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculates recall, accuracy, precision, and F1-score for classification tasks.

        For multi-class classification, metrics are calculated with 'weighted' averaging.
        Handles cases with only one unique class in targets by setting recall, precision,
        and f1-score to 0.0 to avoid errors and warnings.

        Args:
            predictions (torch.Tensor): A 1D tensor of predicted class labels.
            targets (torch.Tensor): A 1D tensor of true class labels.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics:
                              'recall', 'accuracy', 'precision', and 'f1_score'.
        """
        predictions_np: pd.Series = pd.Series(predictions.cpu().numpy())
        targets_np: pd.Series = pd.Series(targets.cpu().numpy())

        unique_targets: int = targets_np.nunique()
        if unique_targets > 1:
            average_mode: str = 'weighted'
            recall: float = recall_score(targets_np, predictions_np, average=average_mode, zero_division=0)
            precision: float = precision_score(targets_np, predictions_np, average=average_mode, zero_division=0)
            f1: float = f1_score(targets_np, predictions_np, average=average_mode, zero_division=0) # Added f1_score
            logger.debug(f"Calculated weighted recall, precision, and f1-score (unique targets > 1).")
        else: # Handle cases with only one unique class in targets
            recall: float = 0.0 
            precision: float = 0.0 
            f1: float = 0.0 # Added f1_score
            logger.warning(f"Only 1 unique class ({targets_np.iloc[0]}) in targets. Recall, Precision, and F1-score set to 0.0.")
        
        accuracy: float = accuracy_score(targets_np, predictions_np)
        logger.debug(f"Calculated metrics: Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}.")
        return {"recall": recall, "accuracy": accuracy, "precision": precision, "f1_score": f1} # Return f1_score

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
            # Removed redundant logger.debug for each batch
            # logger.debug(f"Batch {batch_idx+1}/{len(data_loader)}: Loss={loss.item():.4f}")

        avg_loss = total_loss / len(data_loader.dataset)
        # Removed redundant logger.info for training epoch finished
        # logger.info(f"Training Epoch finished. Average Train Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float, float, float]: # Added f1_score to return type
        """
        Evaluates the model's performance on a given dataset for one epoch.

        Sets the model to evaluation mode, iterates through the data_loader
        without computing gradients, and calculates loss and key metrics.

        Args:
            model (nn.Module): The PyTorch model to be evaluated.
            data_loader (DataLoader): DataLoader containing the evaluation data.
            criterion (nn.Module): The loss function used for evaluation.

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing:
                - avg_loss (float): The average evaluation loss for the epoch.
                - accuracy (float): The accuracy score.
                - recall (float): The recall score.
                - precision (float): The precision score.
                - f1 (float): The F1-score. # Added F1-score
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
                # Removed redundant logger.debug for evaluation batch
                # logger.debug(f"Evaluation Batch {batch_idx+1}/{len(data_loader)}")

        avg_loss: float = total_loss / len(data_loader.dataset)
        
        predictions_tensor: torch.Tensor = torch.tensor(all_predictions)
        targets_tensor: torch.Tensor = torch.tensor(all_targets)

        if len(predictions_tensor) == 0:
            logger.warning("No predictions made during evaluation. Returning 0 for metrics.")
            return avg_loss, 0.0, 0.0, 0.0, 0.0 # Return 0 for f1 as well

        metrics: Dict[str, float] = self._calculate_metrics(predictions_tensor, targets_tensor)
        # Removed redundant logger.info for evaluation epoch finished
        # logger.info(f"Evaluation Epoch finished. Average Test Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return avg_loss, metrics["accuracy"], metrics["recall"], metrics["precision"], metrics["f1_score"] # Return f1_score


    def run_training(
        self,
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        experiment_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrates the full training process for a given model.

        This method iterates through epochs, calling train_epoch and evaluate_epoch,
        applies learning rate scheduling, saves the best model checkpoint, and
        logs detailed epoch-wise results directly to a Parquet file on disk.
        A single rotating backup of the results file is maintained.

        Args:
            model_name (str): De naam van het te trainen model (gebruikt door get_model).
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test/validation dataset.
            experiment_name (str): A unique name for the model being trained (e.g., "Baseline", "CNN_Exp1").
                                   This name is used for logging and file naming.
            model_config (Dict[str, Any]): The specific configuration for this model,
                                           including model-specific hyperparameters.

        Returns:
            Dict[str, Any]: A summary dictionary of the best performing epoch's metrics
                            and overall training run metadata, or an error dictionary if initialization fails.

        Raises:
            TypeError: If train_loader, test_loader, or model_config are of incorrect types.
            ValueError: If model_name or experiment_name are not non-empty strings.
        """
        # --- Input validation and initial setup ---
        logger.info(f"Starting training run for model: {model_name}, experiment: {experiment_name}")
        if not isinstance(train_loader, DataLoader) or not isinstance(test_loader, DataLoader):
            logger.error(f"Invalid DataLoader type provided. train_loader: {type(train_loader)}, test_loader: {type(test_loader)}.")
            # Return an error state to main.py instead of crashing
            return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": "Invalid DataLoader type provided."
            }
        if not isinstance(model_name, str) or not model_name:
            logger.error(f"Invalid model_name: '{model_name}'. Must be a non-empty string.")
            # Return an error state to main.py instead of crashing
            return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": f"Invalid model_name: '{model_name}'. Must be a non-empty string."
            }
        if not isinstance(experiment_name, str) or not experiment_name:
            logger.error(f"Invalid experiment_name: '{experiment_name}'. Must be a non-empty string.")
            # Return an error state to main.py instead of crashing
            return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": f"Invalid experiment_name: '{experiment_name}'. Must be a non-empty string."
            }
        if not isinstance(model_config, dict):
            logger.error(f"Invalid model_config type provided: {type(model_config)}. Expected dictionary.")
            # Return an error state to main.py instead of crashing
            return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": "Invalid model_config type provided. Expected dictionary."
            }

        start_time: datetime = datetime.now()
        
        # --- Model Initialisatie binnen run_training ---
        logger.debug(f"Voorbereiding model '{model_name}' met feature_count={self.feature_count}, class_count={self.class_count}, model_config={model_config}")

        # Create a mutable copy of model_config to potentially modify
        _model_config = model_config.copy()

        if model_name in ["baseline_model", "gru_model"]:
            if _model_config.get("input_size") != self.feature_count:
                logger.warning(f"{model_name} input_size ({_model_config.get('input_size')}) komt niet overeen met het werkelijke aantal features ({self.feature_count}). Aangepast.")
            _model_config["input_size"] = self.feature_count
            
            if _model_config.get("output_size") != self.class_count:
                logger.warning(f"{model_name} output_size ({_model_config.get('output_size')}) komt niet overeen met het werkelijke aantal klassen ({self.class_count}). Aangepast.")
            _model_config["output_size"] = self.class_count
            
        elif model_name == "cnn_model":
            if _model_config.get("output_size") != self.class_count:
                logger.warning(f"CNN model output_size ({_model_config.get('output_size')}) komt niet overeen met het werkelijke aantal klassen ({self.class_count}). Aangepast.")
            _model_config["output_size"] = self.class_count
            
            if "input_size_after_flattening" not in _model_config:
                logger.error("CNN model 'input_size_after_flattening' ontbreekt in model_config. Dit is cruciaal voor de lineaire laag. Zorg dat dit correct wordt doorgegeven.")
                # Return an error state instead of raising ValueError and causing sys.exit(1)
                return {
                    "run_id": self.run_id,
                    "model_type": model_name,
                    "experiment_name": experiment_name,
                    "status": "FAILED",
                    "error_message": "CNN model 'input_size_after_flattening' ontbreekt in model_config."
                }
        
        # Initialize the model using the potentially modified _model_config
        try:
            model = get_model(model_name, _model_config).to(self.device)
            logger.info(f"Model '{model_name}' van experiment '{experiment_name}' geïnitialiseerd en verplaatst naar {self.device}.")
            logger.debug(f"Model architectuur: \n{model}")
            logger.debug(f"Model-specifieke configuratie voor '{model_name}': {_model_config}")
        except Exception as e:
            logger.critical(f"Failed to initialize model '{model_name}' for experiment '{experiment_name}': {e}", exc_info=True)
            # Remove the specific handler for this run's logger before returning error
            if self._log_handler_ids:
                handler_to_remove = self._log_handler_ids.pop()
                try:
                    logger.remove(handler_to_remove)
                    logger.debug(f"Removed specific loguru handler {handler_to_remove} for run {self.run_id}.")
                except ValueError:
                    logger.warning(f"Attempted to remove non-existent or already removed handler {handler_to_remove} for run {self.run_id}.")
                except Exception as e_inner:
                    logger.error(f"Unexpected error removing handler {handler_to_remove}: {e_inner}")
            
            # Return an error state to main.py instead of crashing the entire script
            return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": f"Model initialisatie mislukt: {e}"
            }

        # Loss function with class weights
        criterion: nn.Module = nn.CrossEntropyLoss(weight=self.class_weights) 
        
        # Optimizer setup
        optimizer_name = self.settings["training"]["optimizer"]
        learning_rate = self.settings["training"]["learning_rate"]
        weight_decay = self.settings["training"].get("weight_decay", 0.0)
        
        try:
            optimizer: optim.Optimizer = self._get_optimizer(model, optimizer_name, learning_rate, weight_decay)
        except ValueError as e: # Catch specific optimizer errors
             logger.error(f"Fout bij het initialiseren van de optimizer voor {experiment_name}: {e}")
             return {
                "run_id": self.run_id,
                "model_type": model_name,
                "experiment_name": experiment_name,
                "status": "FAILED",
                "error_message": f"Optimizer initialisatie mislukt: {e}"
            }
        
        # Scheduler setup
        scheduler = None
        if self.settings["training"].get("use_scheduler", False):
            scheduler_type = self.settings["training"].get("scheduler_type", "ReduceLROnPlateau")
            scheduler_kwargs: Dict[str, Any] = self.settings["training"].get("scheduler_kwargs", {})
            try:
                scheduler = self._get_scheduler(optimizer, scheduler_type, scheduler_kwargs)
            except TypeError as e: # Catch specific scheduler errors
                logger.error(f"Fout bij het initialiseren van de scheduler voor {experiment_name}: {e}")
                return {
                    "run_id": self.run_id,
                    "model_type": model_name,
                    "experiment_name": experiment_name,
                    "status": "FAILED",
                    "error_message": f"Scheduler initialisatie mislukt: {e}"
                }
        else:
            logger.info("Scheduler is uitgeschakeld in configuratie.")


        epochs: int = self.settings["training"]["epochs"]
        best_recall: float = float('-inf')
        best_accuracy_for_recall: float = float('-inf') 
        
        best_model_path: str = self.best_model_path_template.format(model_name=experiment_name)
        model_results_parquet_path: str = self.results_path_template.format(model_name=experiment_name)
        model_results_backup_path: str = os.path.join(self.run_log_dir, f"{self.run_id}_{experiment_name}_results.parquet.bak")
        
        logger.debug(f"Model checkpoint zal worden opgeslagen in: {best_model_path}")
        logger.debug(f"Resultaten zullen worden opgeslagen in: {model_results_parquet_path}")
        
        best_epoch_metrics: Dict[str, Any] = {} 

        train_samples = len(train_loader.dataset) if train_loader.dataset is not None else 0
        test_samples = len(test_loader.dataset) if test_loader.dataset is not None else 0

        logger.info(f"Trainingsdataset: {os.path.basename(self.train_data_path)}, Samples: {train_samples}, Batch grootte: {train_loader.batch_size}, Aantal batches: {len(train_loader)}.")
        logger.info(f"Testdataset: {os.path.basename(self.test_data_path)}, Samples: {test_samples}, Batch grootte: {test_loader.batch_size}, Aantal batches: {len(test_loader)}.")


        # --- Training Loop ---
        for epoch in range(1, epochs + 1):
            epoch_start_time: float = time.time()
            
            try:
                train_loss: float = self.train_epoch(model, train_loader, criterion, optimizer)
                test_loss, accuracy_value, recall_value, precision_value, f1_value = self.evaluate_epoch(model, test_loader, criterion)
                epoch_duration: float = time.time() - epoch_start_time

                if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(test_loss)
                elif scheduler and isinstance(scheduler, StepLR):
                    scheduler.step()
                
                loss_verschil: float = train_loss - test_loss
                relatief_verschil_loss: float = (loss_verschil / train_loss) if train_loss != 0 else float('inf')

                logger.info(f"Experiment: {experiment_name}, Epoch: {epoch:{len(str(epochs))}d}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy_value:.4f}, Recall: {recall_value:.4f}, Precision: {precision_value:.4f}, F1: {f1_value:.4f}, Duration: {epoch_duration:.2f}s, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Prepare data for the dataframe
                epoch_data: Dict[str, Any] = {
                    'run_id': self.run_id, 
                    'model_name': experiment_name,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'verschil_loss': loss_verschil,
                    'relatief_verschil_loss': relatief_verschil_loss,
                    'accuracy': accuracy_value,
                    'run_recall': recall_value,
                    'run_precision': precision_value,
                    'run_f1': f1_value, 
                    'duration_epoch_seconds': epoch_duration,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'train_samples': train_samples,
                    'test_samples': test_samples,   
                    **{f'class_{i}_weight': self.class_weights[i].item() for i in range(self.class_count)},
                    **_model_config 
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
                        logger.debug(f"Appended epoch {epoch} data to existing DataFrame for {experiment_name}.")
                    except Exception as e:
                        logger.error(f"Failed to read existing Parquet file {model_results_parquet_path}. Starting new DataFrame. Error: {e}")
                        current_df = pd.DataFrame([epoch_data])
                else:
                    current_df = pd.DataFrame([epoch_data])
                    logger.debug(f"Created new DataFrame for epoch {epoch} results for {experiment_name}.")
                
                try:
                    current_df.to_parquet(model_results_parquet_path, index=False)
                except Exception as e:
                    logger.error(f"Failed to save results for epoch {epoch} to {model_results_parquet_path}. Error: {e}")

                # Save the best model (based on recall, then accuracy)
                if recall_value > best_recall:
                    best_recall = recall_value
                    best_accuracy_for_recall = accuracy_value
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        logger.success(f"Saved best model for {experiment_name} to {best_model_path} (Recall: {best_recall:.4f}, Accuracy: {best_accuracy_for_recall:.4f}).")
                        best_epoch_metrics = { 
                            "best_train_loss": train_loss, 
                            "best_test_loss": test_loss,
                            "best_accuracy": accuracy_value,
                            "best_recall": recall_value,
                            "best_precision": precision_value,
                            "best_f1": f1_value,
                            "best_epoch": epoch
                        }
                    except Exception as e:
                        logger.error(f"Failed to save best model for {experiment_name} to {best_model_path}. Error: {e}")
                elif recall_value == best_recall and accuracy_value > best_accuracy_for_recall:
                    best_accuracy_for_recall = accuracy_value
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        logger.success(f"Saved best model for {experiment_name} to {best_model_path} (Recall: {best_recall:.4f}, Improved Accuracy: {best_accuracy_for_recall:.4f}).")
                        best_epoch_metrics = { 
                            "best_train_loss": train_loss, 
                            "best_test_loss": test_loss,
                            "best_accuracy": accuracy_value,
                            "best_recall": recall_value,
                            "best_precision": precision_value,
                            "best_f1": f1_value,
                            "best_epoch": epoch
                        }
                    except Exception as e:
                        logger.error(f"Failed to save best model for {experiment_name} to {best_model_path}. Error: {e}")
                else:
                    logger.debug(f"Huidige recall {recall_value:.4f} niet beter dan beste {best_recall:.4f} (of nauwkeurigheid niet beter als recall gelijk is). Model niet opgeslagen.")
            
            except Exception as e:
                logger.error(f"Fout tijdens epoch {epoch} voor experiment {experiment_name}: {e}", exc_info=True)
                # Indien een epoch mislukt, kunnen we ervoor kiezen om gewoon door te gaan naar de volgende epoch
                # of de hele training voor dit model als mislukt te markeren en te breken.
                # Voor nu loggen we de fout en gaan we door naar de volgende epoch.
                # Als je de training voor dit model wilt stoppen na een epoch-fout, voeg dan 'break' toe.
                pass # Ga door naar de volgende epoch

        end_time: datetime = datetime.now()
        duration: timedelta = end_time - start_time
        logger.info(f"Training voor model {experiment_name} voltooid. Totale duur: {duration}")

        # Final results dictionary
        final_results: Dict[str, Any] = {
            "run_id": self.run_id, 
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(duration),
            "model_type": model_name,
            "experiment_name": experiment_name,
            "epochs_run": epoch, # Dit zal het laatste epoch-nummer zijn, zelfs als er fouten waren.
            "optimizer": optimizer_name, 
            "learning_rate_initial": learning_rate, 
            "weight_decay": weight_decay, 
            "scheduler_type": self.settings["training"].get("scheduler_type", "None"),
            "scheduler_factor": self.settings["training"]["scheduler_kwargs"].get("factor") if scheduler else None,
            "scheduler_patience": self.settings["training"]["scheduler_kwargs"].get("patience") if scheduler else None,
            "train_data_file": os.path.basename(self.train_data_path), 
            "test_data_file": os.path.basename(self.test_data_path),   
            "train_samples": train_samples, 
            "test_samples": test_samples,   
            **{f'class_{i}_weight': self.class_weights[i].item() for i in range(self.class_count)},
            **best_epoch_metrics, # Deze zal leeg zijn als geen enkele epoch succesvol was
            "model_specific_config": _model_config,
            "status": "COMPLETED" if best_epoch_metrics else "COMPLETED_WITH_ERRORS" # Voeg een status toe
        }
        logger.debug(f"Finale samenvattingsresultaten voor {experiment_name}: {final_results}")

        # BELANGRIJK: Verwijder de specifieke handler die door deze Trainer-instantie is toegevoegd
        if self._log_handler_ids:
            handler_to_remove = self._log_handler_ids.pop()
            try:
                logger.remove(handler_to_remove)
                logger.debug(f"Specifieke Loguru handler {handler_to_remove} verwijderd voor run {self.run_id}.")
            except ValueError:
                logger.warning(f"Poging om niet-bestaande of reeds verwijderde handler {handler_to_remove} te verwijderen voor run {self.run_id}.")
            except Exception as e:
                logger.error(f"Onverwachte fout bij het verwijderen van handler {handler_to_remove}: {e}")

        return final_results
