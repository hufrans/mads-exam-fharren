import os
import toml
from datetime import datetime
import pandas as pd
import torch
import sys
from loguru import logger
import argparse # Importeer argparse

# --- Pad Definities ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..", "..")
sys.path.insert(0, project_root) # Voeg de project root toe aan de PYTHONPATH
logger.debug(f"Project root added to PYTHONPATH: {project_root}")

# Definieer de basis directories voor logs en runs/resultaten
# Deze paden moeten overeenkomen met wat in config.toml staat onder [paths]
# LOG_BASE_DIR is voor algemene logs (main_application.log) EN unieke run-logmappen die de Trainer creëert.
LOG_BASE_DIR = os.path.join(project_root, "logs") 
# RUNS_BASE_DIR is specifiek voor de samenvatting van alle modellen aan het einde.
RUNS_BASE_DIR = os.path.join(project_root, "runs") 

# Zorg ervoor dat de logs en runs directories bestaan
os.makedirs(LOG_BASE_DIR, exist_ok=True)
os.makedirs(RUNS_BASE_DIR, exist_ok=True)

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
           level="INFO", # Houd dit op INFO voor algemene berichten
           rotation="10 MB", 
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

def run_experiment(config_path: str, num_features_actual: int, num_classes_actual: int, current_run_id: str) -> None:
    """
    Hoofdfunctie om het machine learning experiment te orkestreren.
    Laadt data, traint modellen en logt resultaten.

    Args:
        config_path (str): Pad naar het TOML configuratiebestand.
        num_features_actual (int): Het daadwerkelijke aantal features in de dataset.
        num_classes_actual (int): Het daadwerkelijke aantal klassen in de dataset.
        current_run_id (str): De unieke ID voor de huidige run.
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

    # --- Data Laden ---
    logger.info("Start data laden...")
    # Haal de paden voor de trainings- en testdata op uit de configuratie
    train_data_relative_path = config["data"]["train_data_path"]
    test_data_relative_path = config["data"]["test_data_path"]

    # Construeer absolute paden die door get_data_loaders gebruikt kunnen worden
    # Deze paden worden niet veranderd met de run_id, omdat dit de input data betreft.
    train_data_absolute_path = os.path.join(project_root, train_data_relative_path)
    test_data_absolute_path = os.path.join(project_root, test_data_relative_path)
    logger.debug(f"Trainingsdata pad: {train_data_absolute_path}")
    logger.debug(f"Testdata pad: {test_data_absolute_path}")

    logger.info(f"Vastgestelde feature count: {num_features_actual}, class count: {num_classes_actual}")

    # Definieer de namen van je feature kolommen.
    feature_columns = [f"feature_{i}" for i in range(num_features_actual)]
    # Definieer de naam van je target kolom.
    target_column = "target"

    # Defensieve checks: Zorg ervoor dat de modelparameters in de config
    # overeenkomen met de werkelijke data-dimensies. Pas ze aan indien nodig.
    if config["model_params"]["baseline"]["input_size"] != num_features_actual:
        logger.warning(f"Baseline model input_size ({config['model_params']['baseline']['input_size']}) komt niet overeen met het werkelijke aantal features ({num_features_actual}). Aangepast.")
        config["model_params"]["baseline"]["input_size"] = num_features_actual
    if config["model_params"]["baseline"]["output_size"] != num_classes_actual:
        logger.warning(f"Baseline model output_size ({config['model_params']['baseline']['output_size']}) komt niet overeen met het werkelijke aantal klassen ({num_classes_actual}). Aangepast.")
        config["model_params"]["baseline"]["output_size"] = num_classes_actual

    if config["model_params"]["cnn"]["output_size"] != num_classes_actual:
        logger.warning(f"CNN model output_size ({config['model_params']['cnn']['output_size']}) komt niet overeen met het werkelijke aantal klassen ({num_classes_actual}). Aangepast.")
        config["model_params"]["cnn"]["output_size"] = num_classes_actual

    if config["model_params"]["gru"]["output_size"] != num_classes_actual:
        logger.warning(f"GRU model output_size ({config['model_params']['gru']['output_size']}) komt niet overeen met het werkelijke aantal klassen ({num_classes_actual}). Aangepast.")
        config["model_params"]["gru"]["output_size"] = num_classes_actual
    if config["model_params"]["gru"]["input_size"] != num_features_actual:
        logger.warning(f"GRU model input_size ({config['model_params']['gru']['input_size']}) komt niet overeen met het werkelijke aantal features ({num_features_actual}). Aangepast.")
        config["model_params"]["gru"]["input_size"] = num_features_actual

    logger.info(f"Laden van data van {train_data_absolute_path} en {test_data_absolute_path}...")
    # Creëer PyTorch DataLoaders voor training en testen
    try:
        train_loader, test_loader = get_data_loaders(
            train_data_absolute_path, 
            test_data_absolute_path, 
            feature_columns,
            target_column,
            config["training"]["batch_size"]
        )
        logger.success("Data loaders succesvol aangemaakt.")
    except Exception as e:
        logger.error(f"Fout bij het aanmaken van data loaders: {e}")
        sys.exit(1)

    # --- Initialiseer Trainer ---
    logger.info("Initialiseren van de Trainer...")
    # Geef de LOG_BASE_DIR en de UNIEKE RUN_ID door aan de Trainer.
    # De Trainer zal deze RUN_ID gebruiken om zijn eigen logmappen en outputbestanden te benoemen.
    trainer = Trainer(config, num_features_actual, num_classes_actual, 
                      log_base_dir=LOG_BASE_DIR,
                      run_id=current_run_id) 
    all_experiment_results = [] # Lijst om resultaten van alle modelruns te verzamelen
    logger.info("Trainer succesvol geïnitialiseerd.")

    # --- Baseline Model Training ---
    logger.info("\n--- Start Training Baseline Model ---")
    baseline_config = config["model_params"]["baseline"]
    baseline_results = trainer.run_training(
        model_name="baseline_model", # Geef de naam van het model door
        train_loader=train_loader,
        test_loader=test_loader,
        experiment_name="Baseline", # Modelnaam voor logging
        model_config=baseline_config # Hyperparameters voor logging
    )
    all_experiment_results.append(baseline_results)
    logger.info("Baseline model training voltooid.")

    # --- CNN Model Training (Voorbeeld van Hyperparameter Tuning) ---
    logger.info("\n--- Start Training CNN Model (Hyperparameter Tuning) ---")
    cnn_base_config = config["model_params"]["cnn"].copy() # Kopieer om originele config niet te wijzigen
    cnn_base_config["output_size"] = num_classes_actual 
    cnn_base_config["input_channels"] = 1 # Blijft 1 voor 1D numerieke features

    # Definieer de zoekruimte voor hyperparameter tuning voor de CNN.
    cnn_hyper_params_list = [
        {"hidden_size": 32, "conv_filters": [16, 32], "kernel_size": 3, "use_dropout": False},
        {"hidden_size": 64, "conv_filters": [32, 64], "kernel_size": 5, "use_dropout": True, "dropout_rate": 0.3},
        # Voeg hier meer combinaties toe voor uitgebreidere tuning, indien gewenst
    ]

    # Itereer door de gedefinieerde hyperparameter-combinaties voor de CNN
    for i, cnn_params in enumerate(cnn_hyper_params_list):
        logger.info(f"\n--- Uitvoeren CNN Experiment {i+1}/{len(cnn_hyper_params_list)} ---")
        current_cnn_config = {**cnn_base_config, **cnn_params}
        
        # Dynamische berekening van input_size_after_flattening voor CNN
        current_length = num_features_actual 
        conv_filters_for_this_experiment = current_cnn_config.get("conv_filters")

        # Bereken de gereduceerde lengte na elke MaxPool1d laag
        for _ in range(len(conv_filters_for_this_experiment)): 
            current_length = int(current_length / 2) 
            
        # De input_size_after_flattening is de laatste filtergrootte vermenigvuldigd met de resterende lengte
        input_size_after_flattening = int(current_length * conv_filters_for_this_experiment[-1])
        
        current_cnn_config["input_size_after_flattening"] = input_size_after_flattening
        logger.info(f"Berekende CNN input_size_after_flattening voor lineaire laag: {input_size_after_flattening}")
        
        cnn_results = trainer.run_training(
            model_name="cnn_model", # Geef de naam van het model door
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=f"CNN_Experiment_{i+1}", 
            model_config=current_cnn_config 
        )
        all_experiment_results.append(cnn_results)
    logger.info("CNN model training voltooid.")

    # --- GRU Model Training (Voorbeeld van Hyperparameter Tuning) ---
    logger.info("\n--- Start Training GRU Model (Hyperparameter Tuning) ---")
    gru_base_config = config["model_params"]["gru"].copy() 
    gru_base_config["output_size"] = num_classes_actual 
    gru_base_config["input_size"] = num_features_actual 

    # Definieer de zoekruimte voor hyperparameter tuning voor de GRU.
    gru_hyper_params_list = [
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
        # Voeg hier meer combinaties toe voor uitgebreidere tuning, indien gewenst
    ]

    # Itereer door de gedefinieerde hyperparameter-combinaties voor de GRU
    for i, gru_params in enumerate(gru_hyper_params_list):
        logger.info(f"\n--- Uitvoeren GRU Experiment {i+1}/{len(gru_hyper_params_list)} ---")
        current_gru_config = {**gru_base_config, **gru_params}
        gru_results = trainer.run_training(
            model_name="gru_model", # Geef de naam van het model door
            train_loader=train_loader,
            test_loader=test_loader,
            experiment_name=f"GRU_Experiment_{i+1}", 
            model_config=current_gru_config 
        )
        all_experiment_results.append(gru_results)
    logger.info("GRU model training voltooid.")

    # Optioneel: Sla een samenvatting op van de eindresultaten van alle modelruns
    summary_df = pd.DataFrame(all_experiment_results)
    
    # Sla de samenvatting op in de runs_base_dir met de run-ID vooraan
    summary_output_path = os.path.join(RUNS_BASE_DIR, f"{current_run_id}_all_model_summary.parquet")

    try:
        summary_df.to_parquet(summary_output_path, index=False)
        logger.success(f"Alle experiment samenvattingen opgeslagen naar {summary_output_path}")
    except Exception as e:
        logger.error(f"Fout bij het opslaan van de samenvatting van alle experimenten: {e}")

    logger.info("\n--- Einde Experiment ---")

# Dit blok wordt alleen uitgevoerd wanneer het script direct wordt aangeroepen (niet bij importeren als module)
if __name__ == "__main__":
    # Defineer het pad naar het configuratiebestand
    config_file_path = os.path.join(project_root, "config.toml")
    
    # --- Argument parser setup ---
    parser = argparse.ArgumentParser(description="Run machine learning experiments.")
    parser.add_argument(
        "--dummy", 
        action="store_true", # Dit betekent dat als --dummy aanwezig is, het True is, anders False
        help="Use dummy data for training and testing instead of real data. Default is False (use real data)."
    )
    args = parser.parse_args()

    # Laad de config om de verwachte datapad-variabelen te achterhalen voor dummy data generatie
    try:
        temp_config = load_config(config_file_path)
        logger.info(f"Configuratie geladen vanuit {config_file_path}.")
    except Exception as e:
        logger.error(f"Fout bij het laden van configuratie: {e}")
        sys.exit(1)

    # Definieer de werkelijke aantallen features en klassen in je dataset.
    # Deze moeten overeenkomen met je echte dataset of de dummy data die je genereert.
    num_features_actual = 192 
    num_classes_actual = 5
    
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True) 
    logger.info(f"Zorgen dat de data directory bestaat: {data_dir}")

    # Hier wordt gecheckt of de --dummy flag is gebruikt
    if args.dummy:
        logger.info("--- DUMMY DATA MODUS GEACTIVEERD ---")
        logger.info(f"Dummy data parameters: features={num_features_actual}, classes={num_classes_actual}")
        logger.info("Dummy data wordt aangemaakt voor demonstratie. Dit zal de configuratie overschrijven met dummy paden.")

        train_data_filename = temp_config["data"]["train_data_path"].split('/')[-1]
        test_data_filename = temp_config["data"]["test_data_path"].split('/')[-1]

        train_parquet_path_for_dummy = os.path.join(data_dir, train_data_filename)
        test_parquet_path_for_dummy = os.path.join(data_dir, test_data_filename)
        logger.debug(f"Dummy training data pad: {train_parquet_path_for_dummy}")
        logger.debug(f"Dummy test data pad: {test_parquet_path_for_dummy}")
        
        try:
            # Genereer dummy trainingsdata (100 voorbeelden)
            dummy_train_data = {f"feature_{i}": torch.rand(100).tolist() for i in range(num_features_actual)}
            dummy_train_data["target"] = torch.randint(0, num_classes_actual, (100,)).tolist()
            pd.DataFrame(dummy_train_data).to_parquet(train_parquet_path_for_dummy, index=False)

            # Genereer dummy testdata (50 voorbeelden)
            dummy_test_data = {f"feature_{i}": torch.rand(50).tolist() for i in range(num_features_actual)}
            dummy_test_data["target"] = torch.randint(0, num_classes_actual, (50,)).tolist()
            pd.DataFrame(dummy_test_data).to_parquet(test_parquet_path_for_dummy, index=False)
            logger.success(f"Dummy data aangemaakt als Parquet in '{train_parquet_path_for_dummy}' en '{test_parquet_path_for_dummy}'")
        except Exception as e:
            logger.error(f"Fout bij het genereren van dummy data: {e}")
            sys.exit(1)
        
        # Omdat dummy data altijd dezelfde namen heeft, wijzen we de config paden naar deze dummy bestanden
        temp_config["data"]["train_data_path"] = os.path.relpath(train_parquet_path_for_dummy, project_root)
        temp_config["data"]["test_data_path"] = os.path.relpath(test_parquet_path_for_dummy, project_root)

    else:
        logger.info("--- STANDAARD MODUS: GEBRUIK ECHTE DATA ---")
        logger.info(f"Gebruik van data uit configuratiebestand: '{temp_config['data']['train_data_path']}' en '{temp_config['data']['test_data_path']}'.")
    
    # Start de experimentele run met het opgegeven configuratiebestand en de unieke RUN_ID
    try:
        run_experiment(config_file_path, num_features_actual, num_classes_actual, RUN_ID)
        logger.info("Hoofdscript succesvol voltooid.")
    except Exception as e:
        logger.critical(f"Een kritieke fout is opgetreden in het hoofdscript: {e}", exc_info=True)
        sys.exit(1)

