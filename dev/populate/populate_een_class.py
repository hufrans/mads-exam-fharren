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

# --- 0. Configuratie Parameters ---
# Pas de waarden hieronder aan op basis van je dataset en voorkeuren.

# Data Locatie & Specificaties
# Pad naar het Parquet-bestand van de *specifieke minderheidsklasse* die je wilt oversamplen in deze run.
# Bijvoorbeeld: 'data/class_1_ecgs.parquet'
PARQUET_FILE_PATH: str = 'data/split/class_3_ecgs.parquet' 

# De ID van de klasse die in het bovenstaande Parquet-bestand zit. Dit wordt gebruikt voor de output bestandsnamen.
# Bijvoorbeeld: 1, 2, 3, of 4, afhankelijk van je klassen.
CURRENT_CLASS_ID: Union[int, str] = 3 

# De naam van de kolom in je Parquet-bestanden die de classificatie (target) bevat.
TARGET_COLUMN_NAME: str = 'target' 

# De verwachte lengte van elk ECG-segment (aantal numerieke kolommen in je Parquet-bestand, exclusief de target-kolom).
# Voor kolommen "0" t/m "186" is dit 187.
ECG_SEGMENT_LENGTH: int = 187

# GAN Model Hyperparameters
LATENT_DIM: int = 100          # Dimensie van de ruisvector voor de generator
BATCH_SIZE: int = 64
EPOCHS: int = 5000             # Aantal training-epochs (hoger voor echte data, begin conservatief)
BUFFER_SIZE: int = 10000       # Voor het shuffelen van de dataset (minimaal gelijk aan je dataset grootte)

# Output Configuratie
PLOT_INTERVAL: int = 500       # Plot gegenereerde ECG's elke X epochs
num_synthetic_to_generate: int = 35000 # Aantal synthetische samples dat na training wordt gegenereerd

# Hoofdmap voor alle gegenereerde output (plots en parquet)
BASE_OUTPUT_DIR: str = 'gan_generated_output' 
LOG_DIR: str = 'logging'                     # Map voor logbestanden

# --- Learning Rate Configuratie (Vast) ---
INITIAL_GENERATOR_LR: float = 1e-4
INITIAL_DISCRIMINATOR_LR: float = 1e-4

# --- Early Stopping Configuratie ---
# Referentiewaarde voor evenwichtige GAN loss: log(2)
LN_2: float = math.log(2) 
EARLY_STOPPING_MONITOR_EPOCHS: int = 500 # Aantal opeenvolgende epochs om te monitoren voor stabiliteit of oploop
GEN_LOSS_TOLERANCE: float = 0.05         # Hoe dicht gen_loss bij LN_2 moet zijn (bijv. 0.693 +/- 0.05)
DISC_LOSS_MAX_THRESHOLD: float = 0.3     # Maximale waarde voor discriminator loss om als 'laag' te worden beschouwd.
                                         # Let op: lager dan LN_2, maar niet té laag (wat mode collapse kan betekenen).
GEN_LOSS_INCREASE_THRESHOLD: float = 0.02 # De minimale toename van gen_loss over EARLY_STOPPING_MONITOR_EPOCHS 
                                          # om te stoppen (bijv. 0.02 betekent 2% toename van het gemiddelde).
MIN_EPOCHS_BEFORE_EARLY_STOP: int = 3000  # Minimum aantal epochs voordat early stopping actief wordt

# --- Einde Configuratie Parameters ---


# --- Loguru Configuratie (initialisatie) ---
# Verwijder standaard handler zodat we deze zelf kunnen configureren
logger.remove()

# Controleer of de logging directory bestaat, zo niet, maak deze aan
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    logger.info(f"Logging directory '{LOG_DIR}' aangemaakt.")
else:
    logger.info(f"Logging directory '{LOG_DIR}' bestaat al.")

# Voeg een console handler toe (info niveau)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    diagnose=False
)
logger.info("Loguru console logger geconfigureerd.")

# De BASE_OUTPUT_DIR wordt nu gecontroleerd/aangemaakt
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logger.info(f"Basis output directory '{BASE_OUTPUT_DIR}' gecontroleerd/aangemaakt.")


# --- 2. JOUW ECHTE ECG DATA LADEN EN VOORBEREIDEN ---
def load_and_prepare_ecg_data_from_parquet(
    file_path: str,
    target_column_name: str,
    ecg_segment_length: int
) -> np.ndarray:
    """
    Laadt ECG-data uit een Parquet-bestand, dat al gefilterd is op één specifieke klasse,
    en bereidt de data voor op GAN-training.

    Deze functie leest het opgegeven Parquet-bestand, controleert de aanwezigheid
    van de target-kolom (en verwijdert deze), valideert de lengte van de ECG-segmenten,
    normaliseert de data naar het bereik [-1, 1], en reshape't deze naar het formaat
    dat geschikt is voor 1D Convolutionele lagen.

    Args:
        file_path (str): Pad naar het Parquet-bestand dat de ECG-data bevat.
                         Dit bestand wordt verondersteld reeds gefilterd te zijn
                         voor een specifieke minderheidsklasse.
        target_column_name (str): De naam van de kolom die de klassificatie bevat.
                                  Deze kolom wordt genegeerd voor de ECG-features.
        ecg_segment_length (int): De verwachte lengte van elk ECG-segment (aantal numerieke kolommen).

    Returns:
        np.ndarray: Een NumPy array van de voorbereide ECG-segmenten,
                    met de shape (aantal_samples, ECG_SEGMENT_LENGTH, 1),
                    genormaliseerd naar het bereik [-1, 1] en van type float32.

    Raises:
        ValueError: Als geen samples voor de klasse worden gevonden of als
                    het aantal numerieke kolommen niet overeenkomt met de verwachte lengte.
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

# --- 3. De Generator Model ---
def build_generator(latent_dim: int, ecg_segment_length: int) -> Model:
    """
    Bouwt het Generator-model voor de GAN.

    De generator neemt een ruisvector als input en genereert hieruit
    synthetische ECG-segmenten. Het gebruikt Dense lagen om de ruis
    op te schalen en Conv1DTranspose lagen om het signaal stapsgewijs
    op te bouwen naar de gewenste lengte.

    Args:
        latent_dim (int): De dimensie van de input ruisvector.
        ecg_segment_length (int): De gewenste lengte van de gegenereerde ECG-segmenten.

    Returns:
        tf.keras.Model: Het gecompileerde Generator-model.
    """
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
    
    # Activatie naar 'tanh' voor output [-1, 1]
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

# --- 4. De Discriminator Model ---
def build_discriminator(ecg_segment_length: int) -> Model:
    """
    Bouwt het Discriminator-model voor de GAN.

    De discriminator is een binair classificatiemodel dat probeert te onderscheiden
    of een gegeven ECG-segment echt is (uit de trainingsdata) of synthetisch
    (gegenereerd door de Generator).

    Args:
        ecg_segment_length (int): De lengte van de ECG-segmenten die de discriminator als input krijgt.

    Returns:
        tf.keras.Model: Het gecompileerde Discriminator-model.
    """
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
    
    # De Discriminator output nog steeds een kans, dus sigmoid blijft hier
    model.add(layers.Dense(1, activation='sigmoid')) 
    logger.debug(f"Discriminator Dense output shape: {model.output_shape}")

    logger.info("Discriminator model succesvol opgebouwd.")
    return model

# --- 5. Verliesfuncties en Optimizers ---
# BinaryCrossentropy verwacht inputs tussen 0 en 1, wat de output van de discriminator is.
cross_entropy: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) 

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """
    Berekent het verlies van de Discriminator.
    Args:
        real_output (tf.Tensor): De output van de discriminator voor echte ECG-segmenten.
        fake_output (tf.Tensor): De output van de discriminator voor synthetische ECG-segmenten.
    Returns:
        tf.Tensor: De totale verlieswaarde voor de discriminator.
    """
    # De discriminator krijgt nu data in [-1,1], maar zijn output is nog steeds een kans [0,1]
    # Dus de loss berekening blijft hetzelfde t.o.v. de output van de discriminator.
    real_loss: tf.Tensor = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss: tf.Tensor = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss: tf.Tensor = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Berekent het verlies van de Generator.
    Args:
        fake_output (tf.Tensor): De output van de discriminator voor synthetische ECG-segmenten.
    Returns:
        tf.Tensor: De verlieswaarde voor de generator.
    """
    # De generator wil dat de discriminator de fake output als echt classificeert (dicht bij 1).
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Initialiseer de optimizers met vaste initiële learning rates
generator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(INITIAL_GENERATOR_LR) 
discriminator_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(INITIAL_DISCRIMINATOR_LR)
logger.info("Verliesfuncties en optimizers gedefinieerd met vaste initiële learning rates.")


# --- 6. De GAN Training Stap ---
@tf.function 
def train_step(ecg_segments: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Voert één trainingsstap uit voor zowel de Generator als de Discriminator.
    Args:
        ecg_segments (tf.Tensor): Een batch van echte ECG-segmenten uit de dataset.
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Een tuple met het verlies van de Generator en het verlies van de Discriminator.
    """
    noise: tf.Tensor = tf.random.normal([BATCH_SIZE, LATENT_DIM]) 

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_ecg_segments: tf.Tensor = generator(noise, training=True) 

        real_output: tf.Tensor = discriminator(ecg_segments, training=True) 
        fake_output: tf.Tensor = discriminator(generated_ecg_segments, training=True) 

        gen_loss: tf.Tensor = generator_loss(fake_output) 
        disc_loss: tf.Tensor = discriminator_loss(real_output, fake_output) 

    gradients_of_generator: List[tf.Tensor] = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator: List[tf.Tensor] = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# --- 7. Visualisatie Functie ---
def generate_and_save_ecgs(model: Model, epoch: int, test_input: tf.Tensor, class_id: Union[int, str], run_plots_dir: str) -> None:
    """
    Genereert een batch synthetische ECG-segmenten en slaat deze op als plot.
    Args:
        model (tf.keras.Model): Het getrainde Generator-model.
        epoch (int): Het huidige training-epoch-nummer (voor de bestandsnaam).
        test_input (tf.Tensor): Een vaste ruisvector om consistente generatie te visualiseren.
        class_id (Union[int, str]): De ID van de klasse die wordt gegenereerd, voor de bestandsnaam.
        run_plots_dir (str): De specifieke timestamped directory voor plots van deze run.
    """
    logger.debug(f"Genereren en opslaan van ECG-plots voor epoch {epoch}, klasse {class_id}...")
    predictions_tanh_range: tf.Tensor = model(test_input, training=False)
    
    # Converteer van [-1, 1] naar [0, 1] voor plotten
    predictions_0_1_range = (predictions_tanh_range + 1) / 2
    
    plt.figure(figsize=(12, 8)) 
    for i in range(predictions_0_1_range.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.plot(predictions_0_1_range[i, :, 0]) # Plot de 0-1 geschaalde data
        plt.axis('off')
        plt.ylim(-0.1, 1.1) # Plot limiet blijft consistent voor 0-1 bereik
    plt.suptitle(f'Gegenereerde ECGs - Klasse {class_id} - Epoch {epoch}', fontsize=16)
    # Plot opslaan in de run-specifieke directory
    plot_filepath: str = os.path.join(run_plots_dir, f'ecg_class_{class_id}_epoch_{epoch:04d}.png')
    plt.savefig(plot_filepath)
    plt.close()
    logger.info(f"Plot opgeslagen: {plot_filepath}")

seed: tf.Tensor = tf.random.normal([16, LATENT_DIM]) 

# --- 8. Hoofd Trainingsloop ---
if __name__ == '__main__':
    # --- Starttijd van het script vastleggen ---
    script_start_time = datetime.now()

    # Creëer een timestamped directory voor deze run's plots
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_PLOTS_DIR: str = os.path.join(BASE_OUTPUT_DIR, f'run_plots_{timestamp}')
    os.makedirs(RUN_PLOTS_DIR, exist_ok=True)
    logger.info(f"Run-specifieke plots directory '{RUN_PLOTS_DIR}' aangemaakt.")

    # Configureer nu de file handler voor Loguru, inclusief de timestamp en klasse-ID
    log_file_name: str = os.path.join(LOG_DIR, f"gan_class_{CURRENT_CLASS_ID}_{timestamp}.log")
    logger.add(
        log_file_name,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        compression="zip",
        enqueue=True,
        diagnose=True
    )
    logger.info(f"Logbestand geconfigureerd: '{log_file_name}'")

    logger.info("Start GAN-training script.")
    logger.info(f"ECG Segment Lengte: {ECG_SEGMENT_LENGTH}")
    logger.info(f"Latente Dimensie: {LATENT_DIM}")
    logger.info(f"Batch Grootte: {BATCH_SIZE}")
    logger.info(f"Aantal Epochs: {EPOCHS}")
    logger.info(f"Data bestand: {PARQUET_FILE_PATH}")
    logger.info(f"Huidige Klasse ID voor training: {CURRENT_CLASS_ID}")
    logger.info(f"Target Kolom Naam: {TARGET_COLUMN_NAME}")
    logger.info(f"Aantal synthetische samples te genereren: {num_synthetic_to_generate}")
    logger.info(f"Initiële Generator LR: {INITIAL_GENERATOR_LR}")
    logger.info(f"Initiële Discriminator LR: {INITIAL_DISCRIMINATOR_LR}")
    logger.info(f"Early Stopping geconfigureerd: monitoren voor {EARLY_STOPPING_MONITOR_EPOCHS} epochs.")
    logger.info(f"- Criterium 1: Gen Loss bij LN_2 (+/- {GEN_LOSS_TOLERANCE}) en Disc Loss onder {DISC_LOSS_MAX_THRESHOLD}.")
    logger.info(f"- Criterium 2: Gemiddelde Gen Loss loopt op met meer dan {GEN_LOSS_INCREASE_THRESHOLD} over {EARLY_STOPPING_MONITOR_EPOCHS} epochs.")
    logger.info(f"- Actief na minimaal {MIN_EPOCHS_BEFORE_EARLY_STOP} epochs.")


    # Initialiseer Generator en Discriminator modellen
    generator: Model = build_generator(LATENT_DIM, ECG_SEGMENT_LENGTH)
    discriminator: Model = build_discriminator(ECG_SEGMENT_LENGTH)

    try:
        real_ecg_data: np.ndarray = load_and_prepare_ecg_data_from_parquet(
            file_path=PARQUET_FILE_PATH,
            target_column_name=TARGET_COLUMN_NAME,
            ecg_segment_length=ECG_SEGMENT_LENGTH
        )
    except ValueError as e:
        logger.critical(f"Kritieke fout bij het laden van data: {e}. Script wordt afgesloten.")
        exit()

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(real_ecg_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    logger.info(f"TensorFlow Dataset gemaakt met {real_ecg_data.shape[0]} samples.")

    logger.info("Start training van de GAN...")
    
    # --- Variabelen voor Early Stopping ---
    gen_loss_history: List[float] = []
    disc_loss_history: List[float] = []
    early_stopping_counter: int = 0 # Teller voor de "stabiliteit" conditie

    for epoch in range(EPOCHS):
        current_gen_losses: List[float] = []
        current_disc_losses: List[float] = []
        
        for batch_idx, ecg_batch in enumerate(dataset):
            g_loss, d_loss = train_step(ecg_batch)
            current_gen_losses.append(g_loss.numpy())
            current_disc_losses.append(d_loss.numpy())

        avg_gen_loss: float = float(np.mean(current_gen_losses))
        avg_disc_loss: float = float(np.mean(current_disc_losses))

        # Voeg toe aan geschiedenis voor early stopping
        gen_loss_history.append(avg_gen_loss)
        disc_loss_history.append(avg_disc_loss)
        
        current_disc_lr = discriminator_optimizer.learning_rate.numpy()
        current_gen_lr = generator_optimizer.learning_rate.numpy() 

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f} | G_LR: {current_gen_lr:.6f}, D_LR: {current_disc_lr:.6f}")


        if (epoch + 1) % PLOT_INTERVAL == 0:
            generate_and_save_ecgs(generator, epoch + 1, seed, CURRENT_CLASS_ID, RUN_PLOTS_DIR)
            logger.debug(f"Plots gegenereerd en opgeslagen voor epoch {epoch+1}.")

        # --- Early Stopping Logica ---
        if (epoch + 1) >= MIN_EPOCHS_BEFORE_EARLY_STOP:
            # We hebben minimaal EARLY_STOPPING_MONITOR_EPOCHS aan data nodig voor beide checks
            if len(gen_loss_history) >= EARLY_STOPPING_MONITOR_EPOCHS:
                recent_gen_losses = gen_loss_history[-EARLY_STOPPING_MONITOR_EPOCHS:]
                recent_disc_losses = disc_loss_history[-EARLY_STOPPING_MONITOR_EPOCHS:]

                mean_recent_gen_loss = np.mean(recent_gen_losses)
                mean_recent_disc_loss = np.mean(recent_disc_losses)

                # Criterium 1: Gen Loss is stabiel rond LN_2 EN Disc Loss is laag (zoals eerder gevraagd)
                gen_is_near_ln2 = abs(mean_recent_gen_loss - LN_2) < GEN_LOSS_TOLERANCE
                disc_is_low = mean_recent_disc_loss < DISC_LOSS_MAX_THRESHOLD

                if gen_is_near_ln2 and disc_is_low:
                    early_stopping_counter += 1
                    logger.debug(f"Early stopping counter (stabiliteit): {early_stopping_counter}/{EARLY_STOPPING_MONITOR_EPOCHS} (Gen Loss avg: {mean_recent_gen_loss:.4f}, Disc Loss avg: {mean_recent_disc_loss:.4f})")
                    if early_stopping_counter >= EARLY_STOPPING_MONITOR_EPOCHS:
                        logger.info(f"\nEarly stopping geactiveerd na {epoch+1} epochs! (Reden: Gen Loss stabiel rond LN_2 en Disc Loss laag)")
                        logger.info(f"Gemiddelde Gen Loss laatste {EARLY_STOPPING_MONITOR_EPOCHS} epochs: {mean_recent_gen_loss:.4f} (target: {LN_2:.3f})")
                        logger.info(f"Gemiddelde Disc Loss laatste {EARLY_STOPPING_MONITOR_EPOCHS} epochs: {mean_recent_disc_loss:.4f} (max drempel: {DISC_LOSS_MAX_THRESHOLD:.3f})")
                        logger.info("LET OP: Deze early stopping is gebaseerd op een 'lage' Disc Loss zoals gevraagd.")
                        logger.info("Een disc loss rond LN_2 is vaak een teken van een beter evenwicht tussen generator en discriminator.")
                        break # Stop de training loop
                else:
                    early_stopping_counter = 0 # Reset de teller als niet aan de stabiliteitsvoorwaarden voldaan is

                # Criterium 2: Generator loss loopt consistent op
                # Deze check vereist minimaal 2 * EARLY_STOPPING_MONITOR_EPOCHS aan geschiedenis
                if len(gen_loss_history) >= 2 * EARLY_STOPPING_MONITOR_EPOCHS:
                    # Neem het venster vóór het 'recente' venster
                    previous_gen_losses = gen_loss_history[-2*EARLY_STOPPING_MONITOR_EPOCHS : -EARLY_STOPPING_MONITOR_EPOCHS]
                    mean_previous_gen_loss = np.mean(previous_gen_losses)

                    # Controleer of het recente gemiddelde significant hoger is dan het vorige gemiddelde
                    if mean_recent_gen_loss > mean_previous_gen_loss + GEN_LOSS_INCREASE_THRESHOLD:
                        logger.info(f"\nEarly stopping geactiveerd na {epoch+1} epochs! (Reden: Gemiddelde Gen Loss loopt op)")
                        logger.info(f"Gemiddelde Gen Loss (laatste {EARLY_STOPPING_MONITOR_EPOCHS} epochs): {mean_recent_gen_loss:.4f}")
                        logger.info(f"Gemiddelde Gen Loss (epochs {2*EARLY_STOPPING_MONITOR_EPOCHS} tot {EARLY_STOPPING_MONITOR_EPOCHS} geleden): {mean_previous_gen_loss:.4f}")
                        logger.info(f"Overschrijding drempel van {GEN_LOSS_INCREASE_THRESHOLD:.4f}")
                        break # Stop de training loop

            else:
                logger.debug(f"Nog niet genoeg epochs voor early stopping monitoring. (Huidige geschiedenis lengte: {len(gen_loss_history)})")
        else:
            logger.debug(f"Early stopping is nog niet actief voor epoch {epoch+1} (min: {MIN_EPOCHS_BEFORE_EARLY_STOP}).")


    logger.info("\nTraining voltooid!")
    logger.info(f"Gegenereerde voorbeeld-ECG's opgeslagen in: {RUN_PLOTS_DIR}") 

    # --- 9. Genereer de uiteindelijke synthetische data en sla op als Parquet ---
    logger.info(f"Genereren van {num_synthetic_to_generate} synthetische ECG's na training voor klasse {CURRENT_CLASS_ID}...")
    synthetic_noise: tf.Tensor = tf.random.normal([num_synthetic_to_generate, LATENT_DIM])
    
    # De output van de generator is nu in het [-1, 1] bereik
    final_synthetic_ecgs_tanh_range: np.ndarray = generator(synthetic_noise, training=False).numpy()

    # Converteer van [-1, 1] naar [0, 1] voor opslag en uiteindelijke plots
    final_synthetic_ecgs_array = (final_synthetic_ecgs_tanh_range + 1) / 2

    # Optionele check om te zien of na de transformatie alles correct is (zou nu altijd 0-1 moeten zijn)
    min_gen_val = np.min(final_synthetic_ecgs_array)
    max_gen_val = np.max(final_synthetic_ecgs_array)
    logger.info(f"Finaal gegenereerde ECG's (na transformatie naar [0,1]) bereik: Min: {min_gen_val:.4f}, Max: {max_gen_val:.4f}")
    
    logger.info(f"Shape van de gefinaliseerde synthetische ECG's array: {final_synthetic_ecgs_array.shape}")

    final_synthetic_ecgs_2d: np.ndarray = final_synthetic_ecgs_array.squeeze(axis=-1)

    ecg_column_names: List[str] = [str(i) for i in range(ECG_SEGMENT_LENGTH)]

    synthetic_df: pd.DataFrame = pd.DataFrame(final_synthetic_ecgs_2d, columns=ecg_column_names)

    synthetic_df[TARGET_COLUMN_NAME] = CURRENT_CLASS_ID

    output_parquet_filename: str = f'synthetic_ecgs_class_{CURRENT_CLASS_ID}_{timestamp}.parquet'
    # Parquet bestand blijft in de BASE_OUTPUT_DIR
    output_parquet_filepath: str = os.path.join(BASE_OUTPUT_DIR, output_parquet_filename)

    try:
        synthetic_df.to_parquet(output_parquet_filepath, index=False)
        logger.info(f"Synthetische ECG's opgeslagen als Parquet-bestand: '{output_parquet_filepath}'")
    except Exception as e:
        logger.error(f"Fout bij opslaan van synthetische data als Parquet: {e}")

    plt.figure(figsize=(12, 8))
    for i in range(min(16, num_synthetic_to_generate)): 
        plt.subplot(4, 4, i+1)
        plt.plot(final_synthetic_ecgs_array[i, :, 0]) # Plot de 0-1 geschaalde data
        plt.axis('off')
        plt.ylim(-0.1, 1.1) # Plot limiet blijft consistent voor 0-1 bereik
    plt.suptitle(f'Finaal gegenereerde synthetische ECGs voor Klasse {CURRENT_CLASS_ID}', fontsize=16)
    # Finale plot ook in de run-specifieke directory
    final_plot_filepath: str = os.path.join(RUN_PLOTS_DIR, f'final_synthetic_ecgs_class_{CURRENT_CLASS_ID}.png')
    plt.savefig(final_plot_filepath)
    plt.close()
    logger.info(f"Finaal gegenereerde ECG's plot opgeslagen in '{final_plot_filepath}'")

    # --- Eindtijd van het script vastleggen en duur loggen ---
    script_end_time = datetime.now()
    total_duration = script_end_time - script_start_time
    logger.info(f"Script voltooid. Totale looptijd: {total_duration}")