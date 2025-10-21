# Predictor - Rete Neurale per Predizione Output Macchinario

Questo modulo contiene una rete neurale per predire i valori di output di un macchinario (es. pressione, temperatura) basandosi sui parametri operativi di input.

## Struttura del Progetto

```
predictor/
├── src/
│   ├── models/          # Definizioni della rete neurale
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── data/            # Dataset e preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── training/        # Script di training
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/           # Utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
├── data/                # Dataset
│   ├── raw/            # Dati grezzi
│   └── processed/      # Dati preprocessati
├── notebooks/           # Jupyter notebooks per esperimenti
├── configs/             # File di configurazione
│   ├── config.yaml
│   └── example_config.py
├── requirements.txt     # Dipendenze Python
├── train.py            # Script principale di training
└── README.md           # Questo file
```

## Framework: PyTorch vs TensorFlow/Keras

### PyTorch
**Vantaggi:**
- Più flessibile e intuitivo
- Debug più facile (modalità eager execution)
- Ampiamente usato in ricerca
- Maggiore controllo sull'architettura

**Svantaggi:**
- Richiede più codice rispetto a Keras
- Curva di apprendimento leggermente più ripida

**Quando usarlo:**
- Architetture complesse o custom
- Progetti di ricerca
- Quando serve massima flessibilità

### TensorFlow/Keras
**Vantaggi:**
- Keras è molto semplice (high-level API)
- Ottimo per deployment in produzione
- TensorFlow Lite per mobile/embedded

**Svantaggi:**
- Meno flessibile per architetture complesse
- Debug più difficile (storicamente)

**Quando usarlo:**
- Prototipazione rapida
- Deployment in produzione
- Quando la semplicità è priorità

### Scelta per questo progetto
Ho scelto **PyTorch** perché:
1. È moderno e intuitivo
2. Ottimo per imparare i concetti di deep learning
3. Flessibile per futuri miglioramenti
4. Ampia community e documentazione

## Installazione

```bash
# Naviga nella cartella predictor
cd predictor

# Installa le dipendenze
pip install -r requirements.txt
```

## Quick Start

### 1. Prepara i tuoi dati

Crea un file CSV con i tuoi dati. Esempio:

```csv
param1,param2,param3,param4,pressione,temperatura,velocita
1.2,3.4,5.6,7.8,120.5,85.2,1500
2.1,4.3,6.5,8.7,125.3,87.1,1520
...
```

Salva il file in `data/raw/machinery_data.csv`

### 2. Configura il modello

Modifica `configs/example_config.py` con i nomi delle tue colonne:

```python
'input_columns': ['param1', 'param2', 'param3', 'param4'],
'output_columns': ['pressione', 'temperatura', 'velocita'],
```

### 3. Esegui il training

```bash
python train.py
```

### 4. Usa il modello per predizioni

```python
import torch
from src.models import MachineryPredictor
from src.data import DataPreprocessor

# Carica il modello
model = MachineryPredictor(input_size=4, hidden_sizes=[64, 32], output_size=3)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

# Carica lo scaler
preprocessor = DataPreprocessor()
preprocessor.load_scalers('checkpoints/scalers.pkl')

# Fai una predizione
new_input = [[1.5, 3.8, 6.2, 8.1]]  # Nuovi parametri operativi
new_input_scaled = preprocessor.transform(new_input)

with torch.no_grad():
    prediction_scaled = model(torch.FloatTensor(new_input_scaled))
    prediction = preprocessor.inverse_transform_output(prediction_scaled.numpy())

print(f"Predizione: {prediction}")
# Output: Predizione: [[122.3, 86.5, 1510.2]]
```

## Concetti Chiave

### Architettura della Rete Neurale

La rete è una **Feedforward Neural Network** con:

1. **Input Layer**: Riceve i parametri operativi
2. **Hidden Layers**: Elabora i dati attraverso trasformazioni non-lineari
3. **Output Layer**: Produce le predizioni

```
Input (10 params) → [128] → [64] → [32] → Output (5 values)
```

### Componenti Importanti

**Loss Function** (Funzione di perdita):
- **MSE** (Mean Squared Error): Penalizza errori grandi
- **MAE** (Mean Absolute Error): Più robusta agli outlier
- **Huber**: Compromesso tra MSE e MAE

**Optimizer**:
- **Adam**: Ottimizzatore adattivo, il più usato

**Dropout**:
- Previene overfitting "spegnendo" neuroni random durante il training

**Early Stopping**:
- Ferma il training se la validation loss non migliora

## Configurazione Modello

### Modello Piccolo (dataset < 1000 campioni)
```python
from src.models import create_small_model

model = create_small_model(input_size=10, output_size=5)
# Hidden layers: [32, 16]
# Dropout: 0.1
```

### Modello Medio (dataset 1000-10000 campioni)
```python
from src.models import create_medium_model

model = create_medium_model(input_size=10, output_size=5)
# Hidden layers: [128, 64, 32]
# Dropout: 0.2
```

### Modello Grande (dataset > 10000 campioni)
```python
from src.models import create_large_model

model = create_large_model(input_size=10, output_size=5)
# Hidden layers: [256, 128, 64, 32]
# Dropout: 0.3
```

## Workflow Completo

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_csv_data, DataPreprocessor, MachineryDataset
from src.models import create_medium_model
from src.training import ModelTrainer
from src.utils import plot_training_history, plot_predictions, calculate_metrics

# 1. Carica i dati
X, y = load_csv_data(
    'data/raw/machinery_data.csv',
    input_columns=['param1', 'param2', 'param3'],
    output_columns=['pressione', 'temperatura']
)

# 2. Preprocessing
preprocessor = DataPreprocessor(scaling_method='standard')
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

# 3. Crea dataset e dataloader
train_dataset = MachineryDataset(X_train_scaled, y_train_scaled)
val_dataset = MachineryDataset(X_val_scaled, y_val_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Crea e traína il modello
model = create_medium_model(input_size=3, output_size=2)
trainer = ModelTrainer(model, learning_rate=0.001, loss_fn='mse')

history = trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    patience=10
)

# 5. Valuta il modello
y_pred = trainer.predict(X_test_scaled)
y_pred_original = preprocessor.inverse_transform_output(y_pred)

metrics = calculate_metrics(y_test, y_pred_original,
                           output_names=['pressione', 'temperatura'])

# 6. Visualizza risultati
plot_training_history(history['train_losses'], history['val_losses'])
plot_predictions(y_test, y_pred_original,
                output_names=['pressione', 'temperatura'])
```

## Metriche di Valutazione

- **MSE**: Mean Squared Error - Media dei quadrati degli errori
- **RMSE**: Root MSE - Radice di MSE (stessa unità dei dati)
- **MAE**: Mean Absolute Error - Media dei valori assoluti degli errori
- **R²**: Coefficiente di determinazione (0-1, meglio è vicino a 1)
- **MAPE**: Mean Absolute Percentage Error - Errore percentuale

## Tips & Best Practices

1. **Normalizza sempre i dati**: Usa StandardScaler o MinMaxScaler
2. **Inizia con un modello piccolo**: Aggiungi complessità solo se necessario
3. **Usa early stopping**: Previene overfitting
4. **Monitora train vs validation loss**: Gap grande = overfitting
5. **Testa su dati mai visti**: Il test set è sacro!
6. **Salva gli scaler**: Necessari per predizioni future
7. **Riproducibilità**: Usa random_seed per risultati consistenti

## Troubleshooting

**Training loss non scende:**
- Aumenta learning rate (0.01 invece di 0.001)
- Aumenta dimensione del modello
- Controlla che i dati siano normalizzati

**Validation loss peggiore di training loss:**
- Normale! È overfitting
- Aumenta dropout rate
- Riduci dimensione modello
- Aggiungi più dati

**Predizioni tutte simili:**
- Modello troppo piccolo
- Learning rate troppo bassa
- Pochi dati di training

## Prossimi Passi

Quando avrai i dati reali del macchinario:

1. Sostituisci i nomi delle colonne in `configs/example_config.py`
2. Aggiusta `input_size` e `output_size` del modello
3. Sperimenta con diversi `hidden_sizes`
4. Prova diverse `loss_functions` (mse, mae, huber)
5. Ottimizza `learning_rate` e `batch_size`

## Risorse per Imparare

**PyTorch:**
- Tutorial ufficiale: https://pytorch.org/tutorials/
- PyTorch in 60 minuti: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

**Concetti:**
- Neural Networks: http://neuralnetworksanddeeplearning.com/
- Deep Learning Book: https://www.deeplearningbook.org/

**Video (Italiano):**
- Cerca "PyTorch tutorial italiano" su YouTube
- Corso di Deep Learning di Andrew Ng (sottotitoli in italiano)
