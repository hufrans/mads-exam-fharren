import os
import toml
from datetime import datetime
import pandas as pd
import torch
import sys
from loguru import logger # Importeer Loguru's logger

# --- Loguru configuratie voor main.py ---
# Verwijder standaard handlers om te voorkomen dat er dubbele logs verschijnen
logger.remove()
# Voeg een handler toe voor console output (INFO niveau en hoger)
logger.add(sys.stderr, level="INFO")
# Optioneel: voeg een handler toe voor een algemeen logbestand voor main.py activiteit
# Let op: de Trainer zal zijn EIGEN specifieke logbestanden aanmaken in zijn log_dir
# Dit 'main.log' is meer voor algemene app-start-up en shutdown berichten.
logger.add("main_application.log", level="INFO", rotation="10 MB", retention="7 days", compression="zip")
# --- Einde Loguru configuratie ---


# Voeg de project root toe aan de PYTHONPATH voor correcte imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..", "..")
sys.path.insert(0, project_root)
logger.debug(f"Project root added to PYTHONPATH: {project_root}")

# Aangepaste imports voor de projectstructuur: src/fh/
from src.fh.data_loader import get_data_loaders
from src.fh.training_framework import Trainer
from src.fh.models.baseline_model import BaselineModel
from src.fh.models.cnn_model import CNNModel
from src.fh.models.gru_model import GRUModel
from src.fh.utils import load_config # Gebruik jouw load_config functie

def run_experiment(config_path: str) -> None:
    """
    Hoofdfunctie om het machine learning experiment te orkestreren.
    Laadt data, traint modellen en logt resultaten.

    Args:
        config_path (str): Pad naar het TOML configuratiebestand.
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
    train_data_absolute_path = os.path.join(project_root, train_data_relative_path)
    test_data_absolute_path = os.path.join(project_root, test_data_relative_path)
    logger.debug(f"Trainingsdata pad: {train_data_absolute_path}")
    logger.debug(f"Testdata pad: {test_data_absolute_path}")

    # BELANGRIJK: Definieer hier het daadwerkelijke aantal features en klassen in je dataset.
    num_features_actual = 192 # Aangepast naar 192
    num_classes_actual = 5    # Jouw aantal unieke klassen (5 separate classes)
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
    # Voor CNN blijft input_channels 1, omdat we 1D-features hebben.

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
    trainer = Trainer(config_path) # Trainer gebruikt ook de config_path om instellingen te laden
    all_experiment_results = [] # Lijst om resultaten van alle modelruns te verzamelen
    logger.info("Trainer succesvol geïnitialiseerd.")

    # --- Baseline Model Training ---
    logger.info("\n--- Start Training Baseline Model ---")
    baseline_config = config["model_params"]["baseline"]
    baseline_model = BaselineModel(baseline_config)
    baseline_results = trainer.run_training(
        baseline_model,
        train_loader,
        test_loader,
        "Baseline", # Modelnaam voor logging
        baseline_config # Hyperparameters voor logging
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
        
        current_length = num_features_actual 
        conv_filters_for_this_experiment = current_cnn_config.get("conv_filters")

        for _ in range(len(conv_filters_for_this_experiment)): 
            current_length = torch.floor(torch.tensor(current_length / 2)).item()
            
        input_size_after_flattening = int(current_length * conv_filters_for_this_experiment[-1])
        
        current_cnn_config["input_size_after_flattening"] = input_size_after_flattening
        logger.info(f"Berekende CNN input_size_after_flattening voor lineaire laag: {input_size_after_flattening}")
        
        cnn_model = CNNModel(current_cnn_config)
        cnn_results = trainer.run_training(
            cnn_model,
            train_loader,
            test_loader,
            f"CNN_Experiment_{i+1}", 
            current_cnn_config 
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
        gru_model = GRUModel(current_gru_config)
        gru_results = trainer.run_training(
            gru_model,
            train_loader,
            test_loader,
            f"GRU_Experiment_{i+1}", 
            current_gru_config 
        )
        all_experiment_results.append(gru_results)
    logger.info("GRU model training voltooid.")

    # Optioneel: Sla een samenvatting op van de eindresultaten van alle modelruns
    summary_df = pd.DataFrame(all_experiment_results)
    summary_output_path = os.path.join(trainer.log_dir, "all_model_summary.parquet")
    try:
        summary_df.to_parquet(summary_output_path, index=False)
        logger.success(f"Alle experiment samenvattingen opgeslagen naar {summary_output_path}")
    except Exception as e:
        logger.error(f"Fout bij het opslaan van de samenvatting van alle experimenten: {e}")

    logger.info("\n--- Einde Experiment ---")

# Dit blok wordt alleen uitgevoerd wanneer het script direct wordt aangeroepen (niet bij importeren als module)
if __name__ == "__main__":
    logger.info("Start van het hoofdscript.")
    # Bepaal de project root directory voor robuuste padhantering
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.join(current_dir, "..", "..") 
    sys.path.insert(0, project_root)
    logger.debug(f"Project root ingesteld op: {project_root}")

    # Definieer het pad naar het configuratiebestand
    config_file_path = os.path.join(project_root, "config.toml")
    
    # Laad de config om de verwachte datapad-variabelen te achterhalen voor dummy data generatie
    try:
        temp_config = load_config(config_file_path)
        logger.info(f"Tijdelijke configuratie geladen voor dummy data generatie vanuit {config_file_path}.")
    except Exception as e:
        logger.error(f"Fout bij het laden van tijdelijke configuratie voor dummy data: {e}")
        sys.exit(1)

    # Definieer de werkelijke aantallen features en klassen in je dataset.
    num_features_actual = 192 
    num_classes_actual = 5
    logger.debug(f"Dummy data parameters: features={num_features_actual}, classes={num_classes_actual}")

    # Construeer absolute paden voor de dummy data bestanden op basis van de config
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True) 
    logger.info(f"Zorgen dat de data directory bestaat: {data_dir}")

    train_data_filename = temp_config["data"]["train_data_path"].split('/')[-1]
    test_data_filename = temp_config["data"]["test_data_path"].split('/')[-1]

    train_parquet_path_for_dummy = os.path.join(data_dir, train_data_filename)
    test_parquet_path_for_dummy = os.path.join(data_dir, test_data_filename)
    logger.debug(f"Dummy training data pad: {train_parquet_path_for_dummy}")
    logger.debug(f"Dummy test data pad: {test_parquet_path_for_dummy}")
    
    logger.info("--- BELANGRIJK: Kies hier tussen het gebruik van DUMMY DATA of ECHTE DATA ---")
    logger.info("Dummy data wordt aangemaakt voor demonstratie. COMMENTAAR OF VERWIJDER DEZE SECTIE OM JE ECHTE DATABESTANDEN TE GEBRUIKEN.")
    
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

    # --- Einde van de dummy data sectie ---

    # Start de experimentele run met het opgegeven configuratiebestand
    run_experiment(config_file_path)
    logger.info("Hoofdscript succesvol voltooid.")