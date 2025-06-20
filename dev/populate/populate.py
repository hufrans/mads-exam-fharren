"""
Dit script implementeert een 1D Convolutional Generative Adversarial Network (GAN)
om synthetische ECG-segmenten te genereren voor minderheidsklassen in een scheef verdeelde dataset.

Het script is specifiek ontworpen om te werken met Parquet-bestanden,
waarbij elk bestand al gefilterd is om slechts één specifieke doelklasse te bevatten.
De ECG-segmenten worden verwacht als numerieke kolommen (0 t/m 186), gevolgd door een 'target' kolom.

Vereisten:
- tensorflow
- numpy
- matplotlib
- pandas
- pyarrow (voor het lezen van Parquet-bestanden)
- loguru (voor geavanceerde logging)

Installeer benodigde libraries met:
pip install tensorflow numpy matplotlib pandas pyarrow loguru
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Tuple, List, Union
from datetime import datetime 
from loguru import logger
import math # Import math for ln(2)

# --- 0. Configuratie Parameters (Globale Default Waarden) ---
# Deze waarden worden als defaults gebruikt als ze niet per functieaanroep worden overschreven.

# Algemene Data & Model Specificaties
ECG_SEGMENT_LENGTH: int = 187 # De verwachte lengte van elk ECG-segment (aantal numerieke kolommen)
TARGET_COLUMN_NAME: str = 'target' # De naam van de kolom die de classificatie (target) bevat.

# GAN Model Hyperparameters
LATENT_DIM: int = 100          # Dimensie van de ruisvector voor de generator
BATCH_SIZE: int = 64
EPOCHS: int = 5000             # Maximaal aantal training-epochs (als early stopping niet triggert)
BUFFER_SIZE: int = 10000       # Voor het shuffelen van de dataset (minimaal gelijk aan je dataset grootte)

# Output Configuratie
PLOT_INTERVAL: int = 500       # Plot gegenereerde ECG's elke X epochs
BASE_OUTPUT_DIR: str = 'gan_generated_output' # Hoofdmap voor alle gegenereerde output (plots en parquet)
LOG_DIR: str = 'logging'                     # Map voor logbestanden

# Learning Rate Configuratie (Vast)
INITIAL_GENERATOR_LR: float = 1e-4
INITIAL_DISCRIMINATOR_LR: float = 1e-4

# Early Stopping Configuratie
LN_2: float = math.log(2) 
EARLY_STOPPING_MONITOR_EPOCHS: int = 500 # Aantal opeenvolgende epochs om te monitoren voor stabiliteit of oploop
GEN_LOSS_TOLERANCE: float = 0.05         # Hoe dicht gen_loss bij LN_2 moet zijn (bijv. 0.693 +/- 0.05)
DISC_LOSS_MAX_THRESHOLD: float = 0.3     # Maximale waarde voor discriminator loss om als 'laag' te worden beschouwd.
GEN_LOSS_INCREASE_THRESHOLD: float = 0.02 # De minimale toename van gen_loss over EARLY_STOPPING_MONITOR_EPOCHS 
MIN_EPOCHS_BEFORE_EARLY_STOP: int = 3000  # Minimum aantal epochs voordat early stopping actief wordt

# --- Einde Configuratie Parameters ---


# --- Loguru Configuratie (globale setup voor console) ---
logger.remove() # Verwijder standaard handler
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    diagnose=False
)
logger.info("Loguru console logger geconfigureerd.")

# Hoofd output directory wordt globaal gecontroleerd/aangemaakt
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logger.info(f"Basis output directory '{BASE_OUTPUT_DIR}' gecontroleerd/aangemaakt.")
os.makedirs(LOG_DIR, exist_ok=True)
logger.info(f"Logging directory '{LOG_DIR}' gecontroleerd/aangemaakt.")


# --- 1. Data Laden en Voorbereiden ---
def load_and_prepare_ecg_data_from_parquet(
    file_path: str,
    target_column_name: str,
    ecg_segment_length: int
) -> np.ndarray:
    """
    Laadt ECG-data uit een Parquet-bestand en bereidt deze voor op GAN-training.
    """
    logger.info(f"Laden van data uit Parquet-bestand: {file_path}")
    df: pd.DataFrame = pd.read_parquet(file_path)

    logger.info(f"Originele data shape: {df.shape}")
    
    if target_column_name in df.columns:
        unique_classes: np.ndarray = df[target_column_name].unique()
        if len(unique_classes) == 1:
            logger.info(f"Dit bestand bevat data voor klasse: {unique_classes[0]}")
            ecg_features_df: pd.DataFrame = df.drop(columns=[target_column_name])
        else:
            logger.warning(f"Bestand {file_path} bevat meerdere klassen: {unique_classes}. Dit is niet verwacht. Target kolom wordt gedropt.")
            ecg_features_df = df.drop(columns=[target_column_name])
    else:
        logger.warning(f"Kolom '{target_column_name}' niet gevonden in {file_path}. Alle kolommen worden behandeld als ECG-features.")
        ecg_features_df = df.copy()

    if ecg_features_df.shape[1] != ecg_segment_length:
        error_msg = (
            f"Het aantal numerieke kolommen ({ecg_features_df.shape[1]}) komt niet overeen "
            f"met de ingestelde ECG_SEGMENT_LENGTH ({ecg_segment_length}). "
            f"Controleer je Parquet-bestand en de 'ECG_SEGMENT_LENGTH' parameter."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    your_ecg_data: np.ndarray = ecg_features_df.values.astype('float32')

    logger.info(f"Aantal samples geladen: {your_ecg_data.shape[0]}")
    logger.debug(f"Shape van de geladen ECG-features: {your_ecg_data.shape}")

    min_val: float = np.min(your_ecg_data)
    max_val: float = np.max(your_ecg_data)
    if (max_val - min_val) == 0:
        your_ecg_data_normalized: np.ndarray = np.zeros_like(your_ecg_data)
        logger.warning("Alle waarden in de data zijn hetzelfde. Normalisatie resulteert in nullen.")
    else:
        # Normaliseer naar [-1, 1]
        your_ecg_data_normalized = 2 * ((your_ecg_data - min_val) / (max_val - min_val)) - 1

    logger.info(f"Data genormaliseerd tussen {np.min(your_ecg_data_normalized):.2f} en {np.max(your_ecg_data_normalized):.2f}")
    logger.debug(f"Genormaliseerde data shape: {your_ecg_data_normalized.shape}")

    return your_ecg_data_normalized.reshape(-1, ecg_segment_length, 1)

# --- 2. De Generator Model ---
def build_generator(latent_dim: int, ecg_segment_length: int) -> Model:
    """Bouwt het Generator-model voor de GAN."""
    logger.info("Generator model opbouwen...")
    model: tf.keras.Sequential = tf.keras.Sequential()
    
    initial_steps: int = (ecg_segment_length + 7) // 8 

    model.add(layers.Dense(128 * initial_steps, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    logger.debug(f"Generator Dense laag output shape: {model.output_shape}")

    model.add(layers.Reshape((initial_steps, 128)))
    logger.debug(f"Generator Reshape output shape: {model.output_shape}")
    
    model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    logger.debug(f"Generator Conv1DTranspose 1 output shape: {model.output_shape}")

    model.add(layers.Conv1DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    logger.debug(f"Generator Conv1DTranspose 2 output shape: {model.output_shape}")
    
    model.add(layers.Conv1DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh'))
    logger.debug(f"Generator Conv1DTranspose 3 output shape: {model.output_shape}")
    
    if model.output_shape[1] != ecg_segment_length:
        warning_msg = f"Generator output shape ({model.output_shape[1]}) is niet exact {ecg_segment_length}. Zal bijsnijden."
        logger.warning(warning_msg)
        diff: int = model.output_shape[1] - ecg_segment_length
        if diff > 0:
            crop_start: int = diff // 2
            crop_end: int = diff - crop_start
            model.add(layers.Cropping1D(cropping=(crop_start, crop_end)))
            logger.debug(f"Generator Cropping1D output shape: {model.output_shape}")
        else: 
            error_msg = "Fout: Generator output is korter dan gewenste ECG_SEGMENT_LENGTH. Architectuurprobleem."
            logger.error(error_msg)
            raise ValueError(error_msg)

    model.add(layers.Reshape((ecg_segment_length, 1)))
    logger.debug(f"Generator Final Reshape output shape: {model.output_shape}")

    assert model.output_shape == (None, ecg_segment_length, 1)
    logger.info("Generator model succesvol opgebouwd.")
    return model

# --- 3. De Discriminator Model ---
def build_discriminator(ecg_segment_length: int) -> Model:
    """Bouwt het Discriminator-model voor de GAN."""
    logger.info("Discriminator model opbouwen...")
    model: tf.keras.Sequential = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_size=4, strides=2, padding='same', input_shape=[ecg_segment_length, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    logger.debug(f"Discriminator Conv1D 1 output shape: {model.output_shape}")

    model.add(layers.Conv1D(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    logger.debug(f"Discriminator Conv1D 2 output shape: {model.output_shape}")

    model.add(layers.Flatten())
    logger.debug(f"Discriminator Flatten output shape: {model.output_shape}")
    
    model.add(layers.Dense(1, activation='sigmoid')) 
    logger.debug(f"Discriminator Dense output shape: {model.output_shape}")

    logger.info("Discriminator model succesvol opgebouwd.")
    return model

# --- 4. Verliesfuncties en Optimizers ---
cross_entropy: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) 

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Berekent het verlies van de Discriminator."""
    real_loss: tf.Tensor = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss: tf.Tensor = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss: tf.Tensor = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Berekent het verlies van de Generator."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# --- 5. Visualisatie Functie ---
def generate_and_save_ecgs(model: Model, epoch: int, test_input: tf.Tensor, class_id: Union[int, str], run_plots_dir: str) -> None:
    """
    Genereert een batch synthetische ECG-segmenten en slaat deze op als plot.
    """
    logger.debug(f"Genereren en opslaan van ECG-plots voor epoch {epoch}, klasse {class_id}...")
    predictions_tanh_range: tf.Tensor = model(test_input, training=False)
    predictions_0_1_range = (predictions_tanh_range + 1) / 2
    
    plt.figure(figsize=(12, 8)) 
    for i in range(predictions_0_1_range.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.plot(predictions_0_1_range[i, :, 0])
        plt.axis('off')
        plt.ylim(-0.1, 1.1)
    plt.suptitle(f'Gegenereerde ECGs - Klasse {class_id} - Epoch {epoch}', fontsize=16)
    plot_filepath: str = os.path.join(run_plots_dir, f'ecg_class_{class_id}_epoch_{epoch:04d}.png')
    plt.savefig(plot_filepath)
    plt.close()
    logger.info(f"Plot opgeslagen: {plot_filepath}")

# --- 6. De Hoofd Trainings- en Generatiefunctie voor één klasse ---
def train_and_generate_for_class(
    current_class_id: Union[int, str],
    num_synthetic_to_generate: int,
    parquet_file_path: str,
    target_column_name: str,
    ecg_segment_length: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    buffer_size: int,
    plot_interval: int,
    base_output_dir: str,
    log_dir: str,
    initial_generator_lr: float,
    initial_discriminator_lr: float,
    ln_2: float,
    early_stopping_monitor_epochs: int,
    gen_loss_tolerance: float,
    disc_loss_max_threshold: float,
    gen_loss_increase_threshold: float,
    min_epochs_before_early_stop: int
) -> None:
    """
    Traint een GAN om synthetische data te genereren voor een specifieke klasse
    en slaat de resultaten op.
    """
    logger.info(f"Start GAN-training en generatie voor Klasse: {current_class_id}")
    
    # --- Starttijd van deze specifieke run vastleggen ---
    run_start_time = datetime.now()
    timestamp: str = run_start_time.strftime("%Y%m%d_%H%M%S")

    # Configureer run-specifieke output directories
    run_plots_dir: str = os.path.join(base_output_dir, f'run_plots_class_{current_class_id}_{timestamp}')
    os.makedirs(run_plots_dir, exist_ok=True)
    logger.info(f"Run-specifieke plots directory '{run_plots_dir}' aangemaakt voor klasse {current_class_id}.")

    # Configureer run-specifieke logbestand handler
    log_file_name: str = os.path.join(log_dir, f"gan_class_{current_class_id}_{timestamp}.log")
    file_logger_id = logger.add(
        log_file_name,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        compression="zip",
        enqueue=True,
        diagnose=True
    )
    logger.info(f"Logbestand geconfigureerd voor klasse {current_class_id}: '{log_file_name}'")

    logger.info(f"Parameters voor Klasse {current_class_id}:")
    logger.info(f"  Input bestand: {parquet_file_path}")
    logger.info(f"  Aantal synthetische samples te genereren: {num_synthetic_to_generate}")
    logger.info(f"  ECG Segment Lengte: {ecg_segment_length}")
    logger.info(f"  Latente Dimensie: {latent_dim}")
    logger.info(f"  Batch Grootte: {batch_size}")
    logger.info(f"  Max Aantal Epochs: {epochs}")
    logger.info(f"  Plot Interval: {plot_interval}")
    logger.info(f"  Initiële Generator LR: {initial_generator_lr}")
    logger.info(f"  Initiële Discriminator LR: {initial_discriminator_lr}")
    logger.info(f"  Early Stopping: monitoren voor {early_stopping_monitor_epochs} epochs.")
    logger.info(f"  - Criterium 1: Gen Loss bij LN_2 (+/- {gen_loss_tolerance}) en Disc Loss onder {disc_loss_max_threshold}.")
    logger.info(f"  - Criterium 2: Gemiddelde Gen Loss loopt op met meer dan {gen_loss_increase_threshold} over {early_stopping_monitor_epochs} epochs.")
    logger.info(f"  - Actief na minimaal {min_epochs_before_early_stop} epochs.")

    # Initialiseer Generator en Discriminator modellen
    generator: Model = build_generator(latent_dim, ecg_segment_length)
    discriminator: Model = build_discriminator(ecg_segment_length)

    # Initialiseer optimizers
    generator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(initial_generator_lr) 
    discriminator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(initial_discriminator_lr)
    logger.info("Optimizers gedefinieerd.")

    try:
        real_ecg_data: np.ndarray = load_and_prepare_ecg_data_from_parquet(
            file_path=parquet_file_path,
            target_column_name=target_column_name,
            ecg_segment_length=ecg_segment_length
        )
    except ValueError as e:
        logger.critical(f"Kritieke fout bij het laden van data voor klasse {current_class_id}: {e}. Deze run wordt overgeslagen.")
        logger.remove(file_logger_id) # Verwijder de run-specifieke logger
        return # Stop deze functieuitvoering

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(real_ecg_data).shuffle(buffer_size).batch(batch_size)
    logger.info(f"TensorFlow Dataset gemaakt met {real_ecg_data.shape[0]} samples voor klasse {current_class_id}.")

    logger.info(f"Start training van de GAN voor klasse {current_class_id}...")
    
    # Seed voor consistente visualisatie
    seed: tf.Tensor = tf.random.normal([16, latent_dim])

    # Variabelen voor Early Stopping
    gen_loss_history: List[float] = []
    disc_loss_history: List[float] = []
    early_stopping_counter: int = 0 

    for epoch in range(epochs):
        current_gen_losses: List[float] = []
        current_disc_losses: List[float] = []
        
        for batch_idx, ecg_batch in enumerate(dataset):
            # De train_step functie heeft toegang nodig tot de optimizers en modellen
            # We kunnen deze inline definiëren of als nested functions, maar het is handiger
            # om ze direct te gebruiken als ze globaal zijn of als params worden doorgegeven.
            # Voor dit voorbeeld houden we de oorspronkelijke @tf.function structuur.
            # Hiervoor is het essentieel dat generator, discriminator, en hun optimizers
            # buiten deze functie scope beschikbaar zijn, of als argumenten worden doorgegeven.
            # Omdat we ze net hierboven geïnstantieerd hebben, zijn ze beschikbaar.
            noise: tf.Tensor = tf.random.normal([batch_size, latent_dim]) 

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_ecg_segments: tf.Tensor = generator(noise, training=True) 

                real_output: tf.Tensor = discriminator(ecg_batch, training=True) 
                fake_output: tf.Tensor = discriminator(generated_ecg_segments, training=True) 

                gen_loss: tf.Tensor = generator_loss(fake_output) 
                disc_loss: tf.Tensor = discriminator_loss(real_output, fake_output) 

            gradients_of_generator: List[tf.Tensor] = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator: List[tf.Tensor] = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
            current_gen_losses.append(gen_loss.numpy())
            current_disc_losses.append(disc_loss.numpy())

        avg_gen_loss: float = float(np.mean(current_gen_losses))
        avg_disc_loss: float = float(np.mean(current_disc_losses))

        gen_loss_history.append(avg_gen_loss)
        disc_loss_history.append(avg_disc_loss)
        
        current_disc_lr = discriminator_optimizer.learning_rate.numpy()
        current_gen_lr = generator_optimizer.learning_rate.numpy() 

        logger.info(f"Epoch {epoch+1}/{epochs} (Klasse {current_class_id}), Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f} | G_LR: {current_gen_lr:.6f}, D_LR: {current_disc_lr:.6f}")

        if (epoch + 1) % plot_interval == 0:
            generate_and_save_ecgs(generator, epoch + 1, seed, current_class_id, run_plots_dir)
            logger.debug(f"Plots gegenereerd en opgeslagen voor epoch {epoch+1} van klasse {current_class_id}.")

        # --- Early Stopping Logica ---
        if (epoch + 1) >= min_epochs_before_early_stop:
            if len(gen_loss_history) >= early_stopping_monitor_epochs:
                recent_gen_losses = gen_loss_history[-early_stopping_monitor_epochs:]
                recent_disc_losses = disc_loss_history[-early_stopping_monitor_epochs:]

                mean_recent_gen_loss = np.mean(recent_gen_losses)
                mean_recent_disc_loss = np.mean(recent_disc_losses)

                # Criterium 1: Gen Loss is stabiel rond LN_2 EN Disc Loss is laag
                gen_is_near_ln2 = abs(mean_recent_gen_loss - ln_2) < gen_loss_tolerance
                disc_is_low = mean_recent_disc_loss < disc_loss_max_threshold

                if gen_is_near_ln2 and disc_is_low:
                    early_stopping_counter += 1
                    logger.debug(f"Early stopping counter (stabiliteit) voor klasse {current_class_id}: {early_stopping_counter}/{early_stopping_monitor_epochs} (Gen Loss avg: {mean_recent_gen_loss:.4f}, Disc Loss avg: {mean_recent_disc_loss:.4f})")
                    if early_stopping_counter >= early_stopping_monitor_epochs:
                        logger.info(f"\nEarly stopping geactiveerd na {epoch+1} epochs voor klasse {current_class_id}! (Reden: Gen Loss stabiel rond LN_2 en Disc Loss laag)")
                        logger.info(f"Gemiddelde Gen Loss laatste {early_stopping_monitor_epochs} epochs: {mean_recent_gen_loss:.4f} (target: {ln_2:.3f})")
                        logger.info(f"Gemiddelde Disc Loss laatste {early_stopping_monitor_epochs} epochs: {mean_recent_disc_loss:.4f} (max drempel: {disc_loss_max_threshold:.3f})")
                        break # Stop de training loop
                else:
                    early_stopping_counter = 0 

                # Criterium 2: Generator loss loopt consistent op
                if len(gen_loss_history) >= 2 * early_stopping_monitor_epochs:
                    previous_gen_losses = gen_loss_history[-2*early_stopping_monitor_epochs : -early_stopping_monitor_epochs]
                    mean_previous_gen_loss = np.mean(previous_gen_losses)

                    if mean_recent_gen_loss > mean_previous_gen_loss + gen_loss_increase_threshold:
                        logger.info(f"\nEarly stopping geactiveerd na {epoch+1} epochs voor klasse {current_class_id}! (Reden: Gemiddelde Gen Loss loopt op)")
                        logger.info(f"Gemiddelde Gen Loss (laatste {early_stopping_monitor_epochs} epochs): {mean_recent_gen_loss:.4f}")
                        logger.info(f"Gemiddelde Gen Loss (epochs {2*early_stopping_monitor_epochs} tot {early_stopping_monitor_epochs} geleden): {mean_previous_gen_loss:.4f}")
                        logger.info(f"Overschrijding drempel van {gen_loss_increase_threshold:.4f}")
                        break # Stop de training loop

            else:
                logger.debug(f"Nog niet genoeg epochs voor early stopping monitoring voor klasse {current_class_id}. (Huidige geschiedenis lengte: {len(gen_loss_history)})")
        else:
            logger.debug(f"Early stopping is nog niet actief voor epoch {epoch+1} (min: {min_epochs_before_early_stop}) voor klasse {current_class_id}.")


    logger.info(f"Training voltooid voor Klasse {current_class_id}!")
    logger.info(f"Gegenereerde voorbeeld-ECG's voor klasse {current_class_id} opgeslagen in: {run_plots_dir}") 

    # --- Genereer de uiteindelijke synthetische data en sla op als Parquet ---
    logger.info(f"Genereren van {num_synthetic_to_generate} synthetische ECG's na training voor klasse {current_class_id}...")
    synthetic_noise: tf.Tensor = tf.random.normal([num_synthetic_to_generate, latent_dim])
    
    final_synthetic_ecgs_tanh_range: np.ndarray = generator(synthetic_noise, training=False).numpy()
    final_synthetic_ecgs_array = (final_synthetic_ecgs_tanh_range + 1) / 2 # Converteer naar [0, 1]
    
    min_gen_val = np.min(final_synthetic_ecgs_array)
    max_gen_val = np.max(final_synthetic_ecgs_array)
    logger.info(f"Finaal gegenereerde ECG's (na transformatie naar [0,1]) bereik: Min: {min_gen_val:.4f}, Max: {max_gen_val:.4f}")
    
    final_synthetic_ecgs_2d: np.ndarray = final_synthetic_ecgs_array.squeeze(axis=-1)

    ecg_column_names: List[str] = [str(i) for i in range(ecg_segment_length)]

    synthetic_df: pd.DataFrame = pd.DataFrame(final_synthetic_ecgs_2d, columns=ecg_column_names)
    synthetic_df[target_column_name] = current_class_id

    output_parquet_filename: str = f'synthetic_ecgs_class_{current_class_id}_{timestamp}.parquet'
    output_parquet_filepath: str = os.path.join(base_output_dir, output_parquet_filename)

    try:
        synthetic_df.to_parquet(output_parquet_filepath, index=False)
        logger.info(f"Synthetische ECG's opgeslagen als Parquet-bestand: '{output_parquet_filepath}' voor klasse {current_class_id}.")
    except Exception as e:
        logger.error(f"Fout bij opslaan van synthetische data als Parquet voor klasse {current_class_id}: {e}")

    plt.figure(figsize=(12, 8))
    for i in range(min(16, num_synthetic_to_generate)): 
        plt.subplot(4, 4, i+1)
        plt.plot(final_synthetic_ecgs_array[i, :, 0])
        plt.axis('off')
        plt.ylim(-0.1, 1.1)
    plt.suptitle(f'Finaal gegenereerde synthetische ECGs voor Klasse {current_class_id}', fontsize=16)
    final_plot_filepath: str = os.path.join(run_plots_dir, f'final_synthetic_ecgs_class_{current_class_id}.png')
    plt.savefig(final_plot_filepath)
    plt.close()
    logger.info(f"Finaal gegenereerde ECG's plot opgeslagen in '{final_plot_filepath}' voor klasse {current_class_id}.")

    run_end_time = datetime.now()
    run_duration = run_end_time - run_start_time
    logger.info(f"Run voor Klasse {current_class_id} voltooid. Duur: {run_duration}")
    
    logger.remove(file_logger_id) # Verwijder de run-specifieke file logger


# --- 7. Nieuwe Main-functie voor Meerdere Runs ---
if __name__ == '__main__':
    script_overall_start_time = datetime.now()
    logger.info("Script gestart om meerdere GAN-modellen te trainen en te genereren.")

    # Definieer de runs die je wilt uitvoeren
    # Formaat: (klasse_ID, aantal_output_samples, input_parquet_file_path)
    runs_to_execute = [
        # (1, 34012, 'data/split/class_1_ecgs.parquet'),
        # (2, 30447, 'data/split/class_2_ecgs.parquet'),
        # (3, 35594, 'data/split/class_3_ecgs.parquet'),
        (4, 29804, 'data/split/class_4_ecgs.parquet'),
    ]

    for class_id, output_count, input_file in runs_to_execute:
        logger.info(f"\n--- Start verwerking voor Klasse {class_id} ---")
        train_and_generate_for_class(
            current_class_id=class_id,
            num_synthetic_to_generate=output_count,
            parquet_file_path=input_file,
            target_column_name=TARGET_COLUMN_NAME, # Gebruik de globale constante
            ecg_segment_length=ECG_SEGMENT_LENGTH, # Gebruik de globale constante
            latent_dim=LATENT_DIM,                 # Gebruik de globale constante
            batch_size=BATCH_SIZE,                 # Gebruik de globale constante
            epochs=EPOCHS,                         # Gebruik de globale constante
            buffer_size=BUFFER_SIZE,               # Gebruik de globale constante
            plot_interval=PLOT_INTERVAL,           # Gebruik de globale constante
            base_output_dir=BASE_OUTPUT_DIR,       # Gebruik de globale constante
            log_dir=LOG_DIR,                       # Gebruik de globale constante
            initial_generator_lr=INITIAL_GENERATOR_LR, # Gebruik de globale constante
            initial_discriminator_lr=INITIAL_DISCRIMINATOR_LR, # Gebruik de globale constante
            ln_2=LN_2,                             # Gebruik de globale constante
            early_stopping_monitor_epochs=EARLY_STOPPING_MONITOR_EPOCHS, # Gebruik de globale constante
            gen_loss_tolerance=GEN_LOSS_TOLERANCE, # Gebruik de globale constante
            disc_loss_max_threshold=DISC_LOSS_MAX_THRESHOLD, # Gebruik de globale constante
            gen_loss_increase_threshold=GEN_LOSS_INCREASE_THRESHOLD, # Gebruik de globale constante
            min_epochs_before_early_stop=MIN_EPOCHS_BEFORE_EARLY_STOP # Gebruik de globale constante
        )
        logger.info(f"--- Einde verwerking voor Klasse {class_id} ---\n")

    script_overall_end_time = datetime.now()
    overall_duration = script_overall_end_time - script_overall_start_time
    logger.info(f"Alle GAN-trainingen en generaties voltooid. Totale script duur: {overall_duration}")