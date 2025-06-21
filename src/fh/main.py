import os
import toml
from datetime import datetime
import pandas as pd
import torch
import sys

# Voeg de project root toe aan de PYTHONPATH voor correcte imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..", "..")
sys.path.insert(0, project_root)

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
    # Laad de configuratie vanuit het opgegeven TOML-bestand
    config = load_config(config_path)

    print(f"DEBUG: Geladen scheduler_kwargs uit config: {config['training']['scheduler_kwargs']}")

    # --- Data Laden ---
    # Haal de paden voor de trainings- en testdata op uit de configuratie
    # Deze paden in de config zijn relatief, dus maak ze absoluut t.o.v. de project_root
    train_data_relative_path = config["data"]["train_data_path"]
    test_data_relative_path = config["data"]["test_data_path"]

    # Construeer absolute paden die door get_data_loaders gebruikt kunnen worden
    train_data_absolute_path = os.path.join(project_root, train_data_relative_path)
    test_data_absolute_path = os.path.join(project_root, test_data_relative_path)

    # BELANGRIJK: Definieer hier het daadwerkelijke aantal features en klassen in je dataset.
    num_features_actual = 192 # Aangepast naar 192
    num_classes_actual = 5    # Jouw aantal unieke klassen (5 separate classes)

    # Definieer de namen van je feature kolommen.
    feature_columns = [f"feature_{i}" for i in range(num_features_actual)]
    # Definieer de naam van je target kolom.
    target_column = "target"

    # Defensieve checks: Zorg ervoor dat de modelparameters in de config
    # overeenkomen met de werkelijke data-dimensies. Pas ze aan indien nodig.
    if config["model_params"]["baseline"]["input_size"] != num_features_actual:
        print(f"Warning: Baseline model input_size ({config['model_params']['baseline']['input_size']}) does not match actual feature count ({num_features_actual}). Adjusting.")
        config["model_params"]["baseline"]["input_size"] = num_features_actual
    if config["model_params"]["baseline"]["output_size"] != num_classes_actual:
        print(f"Warning: Baseline model output_size ({config['model_params']['baseline']['output_size']}) does not match actual class count ({num_classes_actual}). Adjusting.")
        config["model_params"]["baseline"]["output_size"] = num_classes_actual

    if config["model_params"]["cnn"]["output_size"] != num_classes_actual:
        print(f"Warning: CNN model output_size ({config['model_params']['cnn']['output_size']}) does not match actual class count ({num_classes_actual}). Adjusting.")
        config["model_params"]["cnn"]["output_size"] = num_classes_actual
    # Voor CNN blijft input_channels 1, omdat we 1D-features hebben.

    if config["model_params"]["gru"]["output_size"] != num_classes_actual:
        print(f"Warning: GRU model output_size ({config['model_params']['gru']['output_size']}) does not match actual class count ({num_classes_actual}). Adjusting.")
        config["model_params"]["gru"]["output_size"] = num_classes_actual
    if config["model_params"]["gru"]["input_size"] != num_features_actual:
        print(f"Warning: GRU model input_size ({config['model_params']['gru']['input_size']}) does not match actual feature count ({num_features_actual}). Adjusting.")
        config["model_params"]["gru"]["input_size"] = num_features_actual

    print(f"Laden van data van {train_data_absolute_path} en {test_data_absolute_path}...")
    # Creëer PyTorch DataLoaders voor training en testen
    train_loader, test_loader = get_data_loaders(
        train_data_absolute_path, # Gebruik de absolute paden
        test_data_absolute_path,  # Gebruik de absolute paden
        feature_columns,
        target_column,
        config["training"]["batch_size"]
    )
    print("Data loaders succesvol aangemaakt.")

    # --- Initialiseer Trainer ---
    trainer = Trainer(config_path) # Trainer gebruikt ook de config_path om instellingen te laden
    all_experiment_results = [] # Lijst om resultaten van alle modelruns te verzamelen

    # --- Baseline Model Training ---
    print("\n--- Training Baseline Model ---")
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

    # --- CNN Model Training (Voorbeeld van Hyperparameter Tuning) ---
    print("\n--- Training CNN Model (Hyperparameter Tuning) ---")
    cnn_base_config = config["model_params"]["cnn"].copy() # Kopieer om originele config niet te wijzigen
    cnn_base_config["output_size"] = num_classes_actual # Zorg dat dit overeenkomt met de data
    cnn_base_config["input_channels"] = 1 # Blijft 1 voor 1D numerieke features

    # Definieer de zoekruimte voor hyperparameter tuning voor de CNN.
    cnn_hyper_params_list = [
        {"hidden_size": 32, "conv_filters": [16, 32], "kernel_size": 3, "use_dropout": False},
        {"hidden_size": 64, "conv_filters": [32, 64], "kernel_size": 5, "use_dropout": True, "dropout_rate": 0.3},
        # Voeg hier meer combinaties toe voor uitgebreidere tuning, indien gewenst
    ]

    # Itereer door de gedefinieerde hyperparameter-combinaties voor de CNN
    for i, cnn_params in enumerate(cnn_hyper_params_list):
        print(f"\n--- Uitvoeren CNN Experiment {i+1}/{len(cnn_hyper_params_list)} ---")
        # Combineer de basisconfiguratie met de huidige specifieke hyperparameters
        current_cnn_config = {**cnn_base_config, **cnn_params}
        
        # BELANGRIJK: Bereken dynamisch de 'input_size_after_flattening' voor de lineaire laag van de CNN.
        # Deze berekening moet nu BINNEN de lus plaatsvinden, en zonder de '+1'.
        current_length = num_features_actual # Start met de oorspronkelijke feature lengte (nu 192)
        conv_filters_for_this_experiment = current_cnn_config.get("conv_filters")

        # Simuleer de convolutie- en poolingoperaties om de uiteindelijke lengte te bepalen
        # Er zijn twee MaxPool1d(2) lagen, dus de lengte wordt twee keer gehalveerd.
        for _ in range(len(conv_filters_for_this_experiment)): # <-- HIER IS DE WIJZIGING: VERPLAATST EN '+1' VERWIJDERD
            current_length = torch.floor(torch.tensor(current_length / 2)).item()
            
        # De uiteindelijke grootte na de laatste conv-laag (lengte * aantal filters laatste laag)
        input_size_after_flattening = int(current_length * conv_filters_for_this_experiment[-1]) # Cast naar int
        
        current_cnn_config["input_size_after_flattening"] = input_size_after_flattening
        print(f"Berekende CNN input_size_after_flattening voor lineaire laag: {input_size_after_flattening}") # Dit zal dynamisch zijn
        
        cnn_model = CNNModel(current_cnn_config)
        cnn_results = trainer.run_training(
            cnn_model,
            train_loader,
            test_loader,
            f"CNN_Experiment_{i+1}", # Unieke naam voor deze specifieke run
            current_cnn_config # Parameters van deze run voor logging
        )
        all_experiment_results.append(cnn_results)

    # --- GRU Model Training (Voorbeeld van Hyperparameter Tuning) ---
    print("\n--- Training GRU Model (Hyperparameter Tuning) ---")
    gru_base_config = config["model_params"]["gru"].copy() # Kopieer om originele config niet te wijzigen
    gru_base_config["output_size"] = num_classes_actual # Zorg dat dit overeenkomt met de data
    gru_base_config["input_size"] = num_features_actual # Input size is het aantal features (nu 192)

    # Definieer de zoekruimte voor hyperparameter tuning voor de GRU.
    gru_hyper_params_list = [
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
        # Voeg hier meer combinaties toe voor uitgebreidere tuning, indien gewenst
    ]

    # Itereer door de gedefinieerde hyperparameter-combinaties voor de GRU
    for i, gru_params in enumerate(gru_hyper_params_list):
        print(f"\n--- Uitvoeren GRU Experiment {i+1}/{len(gru_hyper_params_list)} ---")
        # Combineer de basisconfiguratie met de huidige specifieke hyperparameters
        current_gru_config = {**gru_base_config, **gru_params}
        gru_model = GRUModel(current_gru_config)
        gru_results = trainer.run_training(
            gru_model,
            train_loader,
            test_loader,
            f"GRU_Experiment_{i+1}", # Unieke naam voor deze specifieke run
            current_gru_config # Parameters van deze run voor logging
        )
        all_experiment_results.append(gru_results)

    # Optioneel: Sla een samenvatting op van de eindresultaten van alle modelruns
    summary_df = pd.DataFrame(all_experiment_results)
    summary_output_path = os.path.join(trainer.log_dir, "all_model_summary.parquet")
    summary_df.to_parquet(summary_output_path, index=False)
    print(f"\nAlle experiment samenvattingen opgeslagen naar {summary_output_path}")

    print("\n--- Einde Experiment ---")

# Dit blok wordt alleen uitgevoerd wanneer het script direct wordt aangeroepen (niet bij importeren als module)
if __name__ == "__main__":
    # Bepaal de project root directory voor robuuste padhantering
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Dit is src/fh
    project_root = os.path.join(current_dir, "..", "..") # Ga twee directories omhoog naar project_root

    # Definieer het pad naar het configuratiebestand
    config_file_path = os.path.join(project_root, "config.toml")
    
    # Laad de config om de verwachte datapad-variabelen te achterhalen voor dummy data generatie
    # We doen dit hier omdat de dummy data *voordat* run_experiment wordt aangeroepen, geschreven moet worden
    temp_config = load_config(config_file_path)

    # Definieer de werkelijke aantallen features en klassen in je dataset.
    num_features_actual = 192 # Aangepast naar 192 voor dummy data generatie
    num_classes_actual = 5

    # Construeer absolute paden voor de dummy data bestanden op basis van de config
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True) # Zorg ervoor dat de data directory bestaat

    # Gebruik de filenamen uit de config voor dummy data
    train_data_filename = temp_config["data"]["train_data_path"].split('/')[-1]
    test_data_filename = temp_config["data"]["test_data_path"].split('/')[-1]

    train_parquet_path_for_dummy = os.path.join(data_dir, train_data_filename)
    test_parquet_path_for_dummy = os.path.join(data_dir, test_data_filename)
    
    # --- BELANGRIJK: Kies hier tussen het gebruik van DUMMY DATA of ECHTE DATA ---
    #
    # Optie 1: Gebruik DUMMY DATA voor snelle tests of ontwikkeling
    # De onderstaande code genereert synthetische Parquet-bestanden.
    #
    # Optie 2: Gebruik je ECHTE DATA bestanden
    # Als je je ECHTE data wilt gebruiken, ZORG ER DAN VOOR DAT:
    # 1. De bestanden 'heart_big_train.parq' en 'heart_big_test.parq'
    #    daadwerkelijk in de 'data/' map van je project root staan.
    # 2. JE DE VOLGENDE LIJNEN VOOR DUMMY DATA CREATIE COMMENTAAR OF VERWIJDERT!
    
    print("Dummy data wordt aangemaakt voor demonstratie. COMMENTAAR OF VERWIJDER DEZE SECTIE OM JE ECHTE DATABESTANDEN TE GEBRUIKEN.")
    # Genereer dummy trainingsdata (100 voorbeelden)
    dummy_train_data = {f"feature_{i}": torch.rand(100).tolist() for i in range(num_features_actual)}
    dummy_train_data["target"] = torch.randint(0, num_classes_actual, (100,)).tolist()
    pd.DataFrame(dummy_train_data).to_parquet(train_parquet_path_for_dummy, index=False)

    # Genereer dummy testdata (50 voorbeelden)
    dummy_test_data = {f"feature_{i}": torch.rand(50).tolist() for i in range(num_features_actual)}
    dummy_test_data["target"] = torch.randint(0, num_classes_actual, (50,)).tolist()
    pd.DataFrame(dummy_test_data).to_parquet(test_parquet_path_for_dummy, index=False)
    print(f"Dummy data aangemaakt als Parquet in '{train_parquet_path_for_dummy}' en '{test_parquet_path_for_dummy}'")

    # --- Einde van de dummy data sectie ---

    # Start de experimentele run met het opgegeven configuratiebestand
    # run_experiment zal de configuratie opnieuw laden en de absolute paden correct hanteren.
    run_experiment(config_file_path)