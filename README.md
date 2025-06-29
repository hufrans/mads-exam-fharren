# Van Ruwe Data tot Reddingslijn: Hoe AI Hartziekten Voorspelt

Dit project omvat een diepgaand onderzoek naar de toepassing van neurale netwerken voor de vroege detectie en classificatie van hartritmestoornissen op basis van ECG-signalen. Het doel is om een deep learning-model te ontwikkelen dat automatisch kan bepalen of een persoon een normaal of afwijkend hartritme heeft, en bij een afwijking de specifieke klasse van de stoornis kan identificeren.

**Auteur:** Frans Harren
**Studentnummer:** 189165115
**Datum:** 29-06-2025
**GitHub Repository:** [https://github.com/hufrans/mads-exam-fharren.git](https://github.com/hufrans/mads-exam-fharren.git)

## 1. Project Doelstelling

Het hoofddoel van dit onderzoek is het ontwikkelen en evalueren van verschillende deep learning-modellen voor het classificeren van hartritmestoornissen uit ECG-signalen. De focus ligt op het optimaliseren van de 'recall' score, aangezien het in een medische context cruciaal is om zoveel mogelijk zieke gevallen correct te herkennen en zo gemiste diagnoses te minimaliseren.

## 2. Dataset

Dit onderzoek maakt gebruik van de **Physionet's MIT-BIH Arrhythmia Dataset**. Deze dataset bevat 109.446 observaties, onderverdeeld in vijf klassen:

| Klasse | Ziekte                             | Observaties | Percentage |
| ------ | ---------------------------------- | ----------- | ---------- |
| 0      | Normale hartslag                   | 72471       | 82.8%      |
| 1      | Atriale Premature Contractie       | 2588        | 3.0%       |
| 2      | Premature Ventriculaire Contractie | 7338        | 8.4%       |
| 3      | Onbekende ritmes                   | 641         | 0.7%       |
| 4      | Fusiebeat                          | 6590        | 7.5%       |

De dataset is opgesplitst in trainings- en testbestanden (Parquet-formaat) zoals gespecificeerd in `config.toml`. Er wordt ook gebruik gemaakt van synthetische data voor augmentatie.

## 3. Architectuur en Modellen

Binnen dit project zijn de volgende modellen geïmplementeerd en geëvalueerd:

* **Baseline Model (`baseline_model.py`):** Een eenvoudig Multi-Layer Perceptron (MLP) als startpunt.
* **Convolutional Neural Network (CNN) Model (`cnn_model.py`):** Een 1D CNN model, geschikt voor tijdreeksdata zoals ECG-signalen.
* **Gated Recurrent Unit (GRU) Model (`gru_model.py`):** Een recurrent neuraal netwerk (RNN) variant, vaak effectief voor sequentiële data.
* **CNN with Squeeze-and-Excitation (SE) and Skip Connections Model (`cnn_se_skip_model.py`):** Een geavanceerd model dat probeert de prestaties te verbeteren door het integreren van Squeeze-and-Excitation blokken (voor kanaal-attentie) en residuele verbindingen (skip connections) binnen een CNN-structuur.

De modelconfiguraties (aantal lagen, hidden sizes, etc.) worden beheerd via het `config.toml`-bestand.

## 4. Project Structuur

De repository is modulair opgezet voor helderheid en reproduceerbaarheid:

.
├── data/
│   ├── heart_big_train.parq        # Originele trainingsdata
│   ├── heart_big_train_synthetic.parquet # Synthetische trainingsdata (voor augmentatie)
│   └── heart_big_test.parq         # Testdata
├── src/
│   ├── fh/
│   │   ├── models/                 # Modeldefinities
│   │   │   ├── baseline_model.py
│   │   │   ├── cnn_model.py
│   │   │   ├── cnn_se_skip_model.py
│   │   │   └── gru_model.py
│   │   ├── data_loader.py          # Klasse voor data laden en preprocessing
│   │   ├── model_selector.py       # Functie om modellen te selecteren
│   │   ├── training_framework.py   # Algemeen training framework
│   │   └── utils.py                # Hulpprogramma's (bijv. config laden)
│   └── main.py                     # Hoofduitvoerbaar script voor experimenten
├── config.toml                     # Configuratiebestand voor alle experimenten
├── pyproject.toml                  # Project metadata en dependencies
├── README.md                       # Dit bestand
└── runs/                           # Directory voor resultaten van runs (automatisch aangemaakt)
└── logs/                           # Directory voor gedetailleerde logs (automatisch aangemaakt)




## 5. Gebruik

### 5.1. Configuratie

Alle experimentinstellingen worden beheerd via het `config.toml` bestand in de hoofdmap van het project. Hier kun je parameters aanpassen zoals:

* **`[data]`**: Paden naar de training- en testdatasets (inclusief synthetische data).
* **`[training]`**: `batch_size`, `epochs`, `learning_rate`, `optimizer`, `early_stopping_patience`, `use_scheduler`, etc.
* **`[model_params.YOUR_MODEL_NAME]`**: Specifieke parameters voor elk model (e.g., `input_size`, `hidden_size`, `num_layers`, `conv_filters`).

**Belangrijk:** Zorg ervoor dat de `model_name` die je wilt trainen overeenkomt met een gedefinieerde sectie in `[model_params]` in `config.toml`.

### 5.2. Uitvoeren van Experimenten

Je kunt experimenten uitvoeren via het `main.py` script. Dit script ondersteunt het specificeren van het te trainen model en de configuratie via de command-line.

**Syntax:**

```bash
python src/main.py [--dummy_data]
