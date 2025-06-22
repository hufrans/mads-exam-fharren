import os
import toml
from datetime import datetime
import pandas as pd
import torch
import sys
from loguru import logger
import argparse # Importeer argparse
from sklearn.utils import class_weight # Importeer class_weight
import numpy as np # Importeer numpy
from typing import List # Importeer List uit typing

# --- Pad Definities ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..", "..")
sys.path.insert(0, project_root) # Voeg de project root toe aan de PYTHONPATH
logger.debug(f"Project root added to PYTHONPATH: {project_root}")

# Definieer de basis directories voor logs en runs/resultaten
# Deze paden moeten overeenkomen met wat in config.toml staat onder [paths]
# LOG_BASE_DIR is voor algemene logs (main_application.log) EN unieke run-logmappen die de Trainer creëert.
LOG_BASE_DIR = os.path.join(project_root, "logs") 
# RUNS_BASE_dir is specifiek voor de samenvatting van alle modellen aan het einde.
RUNS_BASE_dir = os.path.join(project_root, "runs") 

# Zorg ervoor dat de logs en runs directories bestaan
os.makedirs(LOG_BASE_DIR, exist_ok=True)
os.makedirs(RUNS_BASE_dir, exist_ok=True) 

# --- Genereer een unieke run-ID voor deze executie ---
# Deze ID wordt gebruikt voor alle output bestandsnamen en mappen die bij deze run horen.
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

# --- Loguru Configuratie voor het hoofdscript (main.py) ---
logger.remove() # Verwijder standaard handlers om te voorkomen dat er dubbele logs verschijnen
# Voeg EENMALIG een handler toe voor console output (INFO niveau en hoger, met kleur)
logger.add(sys.stderr, level="INFO", colorize=True, enqueue=True) # enqueue=True voor thread-safe logging

# Voeg een handler toe voor een algemeen logbestand voor main.py activiteit met run-ID
main_log_file_path = os.path.join(LOG_BASE_DIR, f"{RUN_ID}_main_application.log")
logger.add(main_log_file_path, 
           level="DEBUG", # Houd dit op INFO voor algemene berichten
           rotation="100 MB", 
           retention="7 days", 
           compression="zip",
           serialize=False, # Geen JSON formaat voor leesbaarheid
           enqueue=True) # enqueue=True voor thread-safe logging
logger.info(f"Start van het hoofdscript. Algemene logs worden opgeslagen in '{main_log_file_path}'.")
# --- Einde Loguru configuratie ---


# Aangepaste imports voor de projectstructuur: src/fh/
from src.fh.data_loader import get_data_loaders
from src.fh.training_framework import Trainer
from src.fh.utils import load_config # Gebruik jouw load_config functie

def run_experiment(config_path: str, num_features_actual: int, num_classes_actual: int, current_run_id: str, all_train_data_files: List[str], test_data_file: str) -> None:
    """
    Hoofdfunctie om het machine learning experiment te orkestreren.
    Laadt data, traint modellen en logt resultaten.

    Args:
        config_path (str): Pad naar het TOML configuratiebestand.
        num_features_actual (int): Het daadwerkelijke aantal features in de dataset.
        num_classes_actual (int): Het daadwerkelijke aantal klassen in de dataset.
        current_run_id (str): De unieke ID voor de huidige run.
        all_train_data_files (List[str]): Een lijst van absolute bestandsnamen van de trainingsdata.
        test_data_file (str): De absolute bestandsnaam van de testdata.
    """
    logger.info(f"Start van het experiment met configuratiebestand: {config_path}")
    # Laad de configuratie vanuit het opgegeven TOML-bestand
    try:
        config = load_config(config_path)
        logger.info("Configuratiebestand succesvol geladen.")
        logger.debug(f"Geladen scheduler_kwargs uit config: {config['training']['scheduler_kwargs']}")
    except Exception as e:
        logger.error(f"Fout bij het laden van het configuratiebestand {config_path}: {e}")
        sys.exit(1) # Beëindig het script bij een cruciale fout

    all_experiment_results = [] # Lijst om resultaten van alle modelruns te verzamelen
    
    # Definieer de namen van je feature kolommen.
    feature_columns = [str(j) for j in range(num_features_actual)]
    # Definieer de naam van je target kolom.
    target_column = "target"

    # --- NIEUWE VOLGORDE VAN TRAININGEN: Model per model over alle datasets ---

    # 1) Baseline Model Training op alle datasets
    logger.info("\n--- Start Training Baseline Model op alle datasets ---")
    for i, train_data_absolute_path in enumerate(all_train_data_files):
        logger.info(f"\n--- Start Baseline model voor trainingsbestand: {os.path.basename(train_data_absolute_path)} ({i+1}/{len(all_train_data_files)}) ---")
        test_data_absolute_path = test_data_file

        try:
            train_loader, test_loader, train_df, test_df = get_data_loaders( 
                train_data_absolute_path, 
                test_data_absolute_path, 
                feature_columns,
                target_column,
                config["training"]["batch_size"]
            )
            logger.success("Data loaders succesvol aangemaakt.")
            train_targets = train_df[target_column].values
            class_weights_np = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.array(sorted(train_df[target_column].unique())), 
                y=train_targets
            )
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
            logger.info(f"Berekende klasse-gewichten voor {os.path.basename(train_data_absolute_path)}: {class_weights.tolist()}")

            trainer = Trainer(config, num_features_actual, num_classes_actual, 
                              log_base_dir=LOG_BASE_DIR,
                              run_id=current_run_id, 
                              train_data_path=train_data_absolute_path, 
                              test_data_path=test_data_absolute_path,
                              class_weights=class_weights) 
            logger.info("Trainer succesvol geïnitialiseerd.")

            baseline_config_for_run = config["model_params"]["baseline"].copy()
            baseline_config_for_run["input_size"] = num_features_actual
            baseline_config_for_run["output_size"] = num_classes_actual

            baseline_results = trainer.run_training(
                model_name="baseline_model", 
                train_loader=train_loader,
                test_loader=test_loader,
                experiment_name=f"Baseline_{os.path.basename(train_data_absolute_path).split('.')[0]}", 
                model_config=baseline_config_for_run 
            )
            all_experiment_results.append(baseline_results)
            logger.info(f"Baseline model training voltooid voor {os.path.basename(train_data_absolute_path)}.")
        except Exception as e:
            logger.error(f"Fout tijdens Baseline model training voor {os.path.basename(train_data_absolute_path)}: {e}")
            all_experiment_results.append({
                "run_id": current_run_id,
                "train_data_file_used": os.path.basename(train_data_absolute_path),
                "model_name": "baseline_model",
                "status": "FAILED_TRAINING",
                "error_message": f"Fout tijdens training: {e}"
            })
            continue # Ga door met de volgende trainingsbestanden voor dit modeltype

    # 2) GRU Model Training op alle datasets
    logger.info("\n--- Start Training GRU Model op alle datasets ---")
    for i, train_data_absolute_path in enumerate(all_train_data_files):
        logger.info(f"\n--- Start GRU model voor trainingsbestand: {os.path.basename(train_data_absolute_path)} ({i+1}/{len(all_train_data_files)}) ---")
        test_data_absolute_path = test_data_file

        try:
            train_loader, test_loader, train_df, test_df = get_data_loaders( 
                train_data_absolute_path, 
                test_data_absolute_path, 
                feature_columns,
                target_column,
                config["training"]["batch_size"]
            )
            logger.success("Data loaders succesvol aangemaakt.")
            train_targets = train_df[target_column].values
            class_weights_np = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.array(sorted(train_df[target_column].unique())), 
                y=train_targets
            )
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
            logger.info(f"Berekende klasse-gewichten voor {os.path.basename(train_data_absolute_path)}: {class_weights.tolist()}")

            trainer = Trainer(config, num_features_actual, num_classes_actual, 
                              log_base_dir=LOG_BASE_DIR,
                              run_id=current_run_id, 
                              train_data_path=train_data_absolute_path, 
                              test_data_path=test_data_absolute_path,
                              class_weights=class_weights) 
            logger.info("Trainer succesvol geïnitialiseerd.")

            gru_base_config = config["model_params"]["gru"].copy() 
            gru_base_config["output_size"] = num_classes_actual 
            gru_base_config["input_size"] = num_features_actual 

            gru_hyper_params_list = [
                {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
                {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
            ]

            for j, gru_params in enumerate(gru_hyper_params_list):
                logger.info(f"\n--- Uitvoeren GRU Experiment {j+1}/{len(gru_hyper_params_list)} voor {os.path.basename(train_data_absolute_path)} ---")
                current_gru_config_for_run = {**gru_base_config, **gru_params}
                gru_results = trainer.run_training(
                    model_name="gru_model", 
                    train_loader=train_loader,
                    test_loader=test_loader,
                    experiment_name=f"GRU_Experiment_{j+1}_{os.path.basename(train_data_absolute_path).split('.')[0]}", 
                    model_config=current_gru_config_for_run 
                )
                all_experiment_results.append(gru_results)
            logger.info(f"GRU model training voltooid voor {os.path.basename(train_data_absolute_path)}.")
        except Exception as e:
            logger.error(f"Fout tijdens GRU model training voor {os.path.basename(train_data_absolute_path)}: {e}")
            all_experiment_results.append({
                "run_id": current_run_id,
                "train_data_file_used": os.path.basename(train_data_absolute_path),
                "model_name": "gru_model",
                "status": "FAILED_TRAINING",
                "error_message": f"Fout tijdens training: {e}"
            })
            continue # Ga door met de volgende trainingsbestanden voor dit modeltype


    # 3) CNN Model Training op alle datasets
    logger.info("\n--- Start Training CNN Model op alle datasets ---")
    for i, train_data_absolute_path in enumerate(all_train_data_files):
        logger.info(f"\n--- Start CNN model voor trainingsbestand: {os.path.basename(train_data_absolute_path)} ({i+1}/{len(all_train_data_files)}) ---")
        test_data_absolute_path = test_data_file

        try:
            train_loader, test_loader, train_df, test_df = get_data_loaders( 
                train_data_absolute_path, 
                test_data_absolute_path, 
                feature_columns,
                target_column,
                config["training"]["batch_size"]
            )
            logger.success("Data loaders succesvol aangemaakt.")
            train_targets = train_df[target_column].values
            class_weights_np = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.array(sorted(train_df[target_column].unique())), 
                y=train_targets
            )
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
            logger.info(f"Berekende klasse-gewichten voor {os.path.basename(train_data_absolute_path)}: {class_weights.tolist()}")

            trainer = Trainer(config, num_features_actual, num_classes_actual, 
                              log_base_dir=LOG_BASE_DIR,
                              run_id=current_run_id, 
                              train_data_path=train_data_absolute_path, 
                              test_data_path=test_data_absolute_path,
                              class_weights=class_weights) 
            logger.info("Trainer succesvol geïnitialiseerd.")

            cnn_base_config = config["model_params"]["cnn"].copy() 
            cnn_base_config["output_size"] = num_classes_actual 
            cnn_base_config["input_channels"] = 1 
            cnn_base_config["num_features"] = num_features_actual 

            cnn_hyper_params_list = [
                {"hidden_size": 32, "conv_filters": [16, 32], "kernel_size": 3, "use_dropout": False},
                {"hidden_size": 64, "conv_filters": [32, 64], "kernel_size": 5, "use_dropout": True, "dropout_rate": 0.3},
            ]

            for j, cnn_params in enumerate(cnn_hyper_params_list):
                logger.info(f"\n--- Uitvoeren CNN Experiment {j+1}/{len(cnn_hyper_params_list)} voor {os.path.basename(train_data_absolute_path)} ---")
                current_cnn_config_for_run = {**cnn_base_config, **cnn_params}
                
                current_length = num_features_actual 
                conv_filters_for_this_experiment = current_cnn_config_for_run.get("conv_filters")

                for _ in conv_filters_for_this_experiment: 
                    current_length = max(1, int(current_length // 2)) 
                    
                input_size_after_flattening = int(current_length * conv_filters_for_this_experiment[-1])
                
                current_cnn_config_for_run["input_size_after_flattening"] = input_size_after_flattening
                logger.info(f"Berekende CNN input_size_after_flattening voor lineaire laag: {input_size_after_flattening}")
                
                cnn_results = trainer.run_training(
                    model_name="cnn_model", 
                    train_loader=train_loader,
                    test_loader=test_loader,
                    experiment_name=f"CNN_Experiment_{j+1}_{os.path.basename(train_data_absolute_path).split('.')[0]}", 
                    model_config=current_cnn_config_for_run 
                )
                all_experiment_results.append(cnn_results)
            logger.info(f"CNN model training voltooid voor {os.path.basename(train_data_absolute_path)}.")
        except Exception as e:
            logger.error(f"Fout tijdens CNN model training voor {os.path.basename(train_data_absolute_path)}: {e}")
            all_experiment_results.append({
                "run_id": current_run_id,
                "train_data_file_used": os.path.basename(train_data_absolute_path),
                "model_name": "cnn_model",
                "status": "FAILED_TRAINING",
                "error_message": f"Fout tijdens training: {e}"
            })
            continue # Ga door met de volgende trainingsbestanden voor dit modeltype


    # 4) CNN_SE_Skip Model Training op alle datasets
    if "cnn_se_skip" in config["model_params"]:
        logger.info("\n--- Start Training CNN_SE_Skip Model op alle datasets ---")
        for i, train_data_absolute_path in enumerate(all_train_data_files):
            logger.info(f"\n--- Start CNN_SE_Skip model voor trainingsbestand: {os.path.basename(train_data_absolute_path)} ({i+1}/{len(all_train_data_files)}) ---")
            test_data_absolute_path = test_data_file

            try:
                train_loader, test_loader, train_df, test_df = get_data_loaders( 
                    train_data_absolute_path, 
                    test_data_absolute_path, 
                    feature_columns,
                    target_column,
                    config["training"]["batch_size"]
                )
                logger.success("Data loaders succesvol aangemaakt.")
                train_targets = train_df[target_column].values
                class_weights_np = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.array(sorted(train_df[target_column].unique())), 
                    y=train_targets
                )
                class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
                logger.info(f"Berekende klasse-gewichten voor {os.path.basename(train_data_absolute_path)}: {class_weights.tolist()}")

                trainer = Trainer(config, num_features_actual, num_classes_actual, 
                                  log_base_dir=LOG_BASE_DIR,
                                  run_id=current_run_id, 
                                  train_data_path=train_data_absolute_path, 
                                  test_data_path=test_data_absolute_path,
                                  class_weights=class_weights) 
                logger.info("Trainer succesvol geïnitialiseerd.")

                cnn_se_skip_base_config = config["model_params"]["cnn_se_skip"].copy() 
                cnn_se_skip_base_config["output_size"] = num_classes_actual 
                cnn_se_skip_base_config["input_channels"] = 1 
                cnn_se_skip_base_config["num_features"] = num_features_actual 

                cnn_se_skip_hyper_params_list = [
                    {"hidden_size": 64, "conv_filters_per_block": 32, "kernel_size_per_block": 3, "num_layers": 2, "reduction_ratio": 8, "use_dropout": False},
                    {"hidden_size": 128, "conv_filters_per_block": 64, "kernel_size_per_block": 5, "num_layers": 3, "reduction_ratio": 16, "use_dropout": True, "dropout_rate": 0.4},
                ]

                for j, cnn_se_skip_params in enumerate(cnn_se_skip_hyper_params_list):
                    logger.info(f"\n--- Uitvoeren CNN_SE_Skip Experiment {j+1}/{len(cnn_se_skip_hyper_params_list)} voor {os.path.basename(train_data_absolute_path)} ---")
                    current_cnn_se_skip_config_for_run = {**cnn_se_skip_base_config, **cnn_se_skip_params}
                    
                    current_length_se = num_features_actual 
                    num_blocks_se = current_cnn_se_skip_config_for_run.get("num_layers")
                    filters_per_block_se = current_cnn_se_skip_config_for_run.get("conv_filters_per_block")

                    for _ in range(num_blocks_se):
                        current_length_se = max(1, int(current_length_se // 2)) 
                        
                    input_size_after_flattening_se = int(current_length_se * filters_per_block_se) 
                    
                    current_cnn_se_skip_config_for_run["input_size_after_flattening"] = input_size_after_flattening_se
                    logger.info(f"Berekende CNN_SE_Skip input_size_after_flattening voor lineaire laag: {input_size_after_flattening_se}")
                    
                    cnn_se_skip_results = trainer.run_training(
                        model_name="cnn_se_skip_model", 
                        train_loader=train_loader,
                        test_loader=test_loader,
                        experiment_name=f"CNN_SE_Skip_Experiment_{j+1}_{os.path.basename(train_data_absolute_path).split('.')[0]}", 
                        model_config=current_cnn_se_skip_config_for_run 
                    )
                    all_experiment_results.append(cnn_se_skip_results)
                logger.info(f"CNN_SE_Skip model training voltooid voor {os.path.basename(train_data_absolute_path)}.")
            except Exception as e:
                logger.error(f"Fout tijdens CNN_SE_Skip model training voor {os.path.basename(train_data_absolute_path)}: {e}")
                all_experiment_results.append({
                    "run_id": current_run_id,
                    "train_data_file_used": os.path.basename(train_data_absolute_path),
                    "model_name": "cnn_se_skip_model",
                    "status": "FAILED_TRAINING",
                    "error_message": f"Fout tijdens training: {e}"
                })
                continue # Ga door met de volgende trainingsbestanden voor dit modeltype
    else:
        logger.info("CNN_SE_Skip model is niet geconfigureerd in config.toml, overslaan van training.")


    # Optioneel: Sla een samenvatting op van de eindresultaten van alle modelruns
    summary_df = pd.DataFrame(all_experiment_results)
    
    summary_output_path = os.path.join(RUNS_BASE_dir, f"{current_run_id}_all_model_summary.parquet") 

    try:
        summary_df.to_parquet(summary_output_path, index=False)
        logger.success(f"Alle experiment samenvattingen opgeslagen naar {summary_output_path}")
    except Exception as e:
        logger.error(f"Fout bij het opslaan van de samenvatting van alle experimenten: {e}")

    logger.info("\n--- Einde Experiment ---")

# Dit blok wordt alleen uitgevoerd wanneer het script direct wordt aangeroepen (niet bij importeren als module)
if __name__ == "__main__":
    config_file_path = os.path.join(project_root, "config.toml")
    
    parser = argparse.ArgumentParser(description="Run machine learning experiments.")
    parser.add_argument(
        "--dummy", 
        action="store_true", 
        help="Use dummy data for training and testing instead of real data. Default is False (use real data)."
    )
    args = parser.parse_args()

    try:
        temp_config = load_config(config_file_path)
        logger.info(f"Configuratie geladen vanuit {config_file_path}.")
    except Exception as e:
        logger.error(f"Fout bij het laden van configuratie: {e}")
        sys.exit(1)

    num_features_actual = 187
    num_classes_actual = 5 
    
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True) 
    logger.info(f"Zorgen dat de data directory bestaat: {data_dir}")

    final_train_data_paths: List[str] = [] 
    final_test_data_path = ""

    if args.dummy:
        logger.info("--- DUMMY DATA MODUS GEACTIVEERD ---")
        logger.info(f"Dummy data parameters: features={num_features_actual}, classes={num_classes_actual}")
        logger.info("Dummy data wordt aangemaakt voor demonstratie. Dit zal de configuratie overschrijven met dummy paden.")

        if "train_data_path" not in temp_config["data"] or not isinstance(temp_config["data"]["train_data_path"], list) or len(temp_config["data"]["train_data_path"]) < 2:
            logger.error("config.toml['data']['train_data_path'] moet een lijst zijn met minstens twee paden voor dummy data generatie.")
            sys.exit(1)

        dummy_train_filename_1 = temp_config["data"]["train_data_path"][0].split('/')[-1] 
        dummy_train_filename_2 = temp_config["data"]["train_data_path"][1].split('/')[-1] 
        test_data_filename = temp_config["data"]["test_data_path"].split('/')[-1]

        train_parquet_path_for_dummy_1 = os.path.join(data_dir, dummy_train_filename_1)
        train_parquet_path_for_dummy_2 = os.path.join(data_dir, dummy_train_filename_2)
        test_parquet_path_for_dummy = os.path.join(data_dir, test_data_filename)
        
        logger.debug(f"Dummy training data pad 1: {train_parquet_path_for_dummy_1}")
        logger.debug(f"Dummy training data pad 2: {train_parquet_path_for_dummy_2}")
        logger.debug(f"Dummy test data pad: {test_parquet_path_for_dummy}")
        
        try:
            dummy_train_data_1 = {str(i): torch.rand(100).tolist() for i in range(num_features_actual)}
            targets_1 = torch.cat([torch.zeros(80, dtype=torch.long), 
                                   torch.randint(1, num_classes_actual, (20,), dtype=torch.long)]).tolist()
            dummy_train_data_1["target"] = targets_1
            pd.DataFrame(dummy_train_data_1).to_parquet(train_parquet_path_for_dummy_1, index=False)

            dummy_train_data_2 = {str(i): torch.rand(120).tolist() for i in range(num_features_actual)}
            targets_2 = torch.randint(0, num_classes_actual, (120,)).tolist()
            dummy_train_data_2["target"] = targets_2
            pd.DataFrame(dummy_train_data_2).to_parquet(train_parquet_path_for_dummy_2, index=False)

            dummy_test_data = {str(i): torch.rand(50).tolist() for i in range(num_features_actual)}
            dummy_test_data["target"] = torch.randint(0, num_classes_actual, (50,)).tolist()
            pd.DataFrame(dummy_test_data).to_parquet(test_parquet_path_for_dummy, index=False)
            logger.success(f"Dummy data aangemaakt als Parquet in '{train_parquet_path_for_dummy_1}', '{train_parquet_path_for_dummy_2}' en '{test_parquet_path_for_dummy}'")
        except Exception as e:
            logger.error(f"Fout bij het genereren van dummy data: {e}")
            sys.exit(1)
        
        final_train_data_paths.extend([train_parquet_path_for_dummy_1, train_parquet_path_for_dummy_2])
        final_test_data_path = test_parquet_path_for_dummy

    else:
        logger.info("--- STANDAARD MODUS: GEBRUIK ECHTE DATA ---")
        if "train_data_path" not in temp_config["data"] or not isinstance(temp_config["data"]["train_data_path"], list) or not temp_config["data"]["train_data_path"]:
            logger.error("config.toml['data']['train_data_path'] is niet correct gedefinieerd. Het moet een niet-lege lijst van paden zijn.")
            sys.exit(1)

        for path in temp_config["data"]["train_data_path"]: 
            final_train_data_paths.append(os.path.join(project_root, path))
        final_test_data_path = os.path.join(project_root, temp_config["data"]["test_data_path"])
        logger.info(f"Gebruik van training data uit configuratiebestand: {final_train_data_paths}")
        logger.info(f"Gebruik van test data uit configuratiebestand: '{final_test_data_path}'.")
    
    try:
        run_experiment(config_file_path, num_features_actual, num_classes_actual, RUN_ID, final_train_data_paths, final_test_data_path)
        logger.info("Hoofdscript succesvol voltooid.")
    except Exception as e:
        logger.critical(f"Een kritieke fout is opgetreden in het hoofdscript: {e}", exc_info=True)
        sys.exit(1)
