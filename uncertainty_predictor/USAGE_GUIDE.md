# Guida all'Uso - Uncertainty Predictor

## 🎯 Installazione (Solo Prima Volta)

```bash
cd uncertainty_predictor
pip install -r requirements.txt
```

---

## 🚀 OPZIONE 1: Training Rapido (Dati Auto-Generati)

**Usa questa opzione per testare velocemente il sistema**

```bash
cd uncertainty_predictor
python train.py
```

✅ Il sistema genera automaticamente 2000 campioni sintetici e fa il training.

---

## 📊 OPZIONE 2: Training con i TUOI Dati

### Step 1: Prepara il CSV

Crea un file CSV con i tuoi dati:

**Esempio: `production_data.csv`**
```csv
temperatura,pressione,velocita,qualita
25.3,101.2,1500,95.2
26.1,102.5,1480,94.8
24.8,100.9,1520,96.1
...
```

### Step 2: Configura

Apri `configs/example_config.py` e modifica:

```python
CONFIG = {
    'data': {
        # IMPORTANTE: Cambia questi valori!
        'csv_path': 'production_data.csv',           # ← Il tuo file CSV
        'input_columns': ['temperatura', 'pressione', 'velocita'],  # ← Features
        'output_columns': ['qualita'],               # ← Target

        # Opzionale: puoi modificare questi
        'scaling_method': 'standard',  # o 'minmax'
        'train_size': 0.7,    # 70% training
        'val_size': 0.15,     # 15% validation
        'test_size': 0.15,    # 15% test
    },

    'training': {
        'epochs': 400,           # Numero di epoche
        'batch_size': 32,
        'learning_rate': 0.0001,
        # ...
    }
}
```

### Step 3: Copia il CSV

```bash
# Metti il tuo CSV nella cartella uncertainty_predictor
cp /percorso/al/tuo/production_data.csv uncertainty_predictor/
```

### Step 4: Training

```bash
cd uncertainty_predictor
python train.py
```

---

## 📈 Risultati del Training

Dopo il training, trovi i risultati in `checkpoints_uncertainty/`:

### File Generati

| File | Descrizione |
|------|-------------|
| `best_model.pth` | Modello addestrato (pesi della rete) |
| `scalers.pkl` | Normalizzatori (per preprocessing) |
| `training_history.json` | Storico del training |
| `training_history.png` | **Grafici loss durante training** |
| `predictions_with_uncertainty.png` | **Predizioni con bande di incertezza** |
| `scatter_with_uncertainty.png` | **Scatter plot predetto vs reale** |
| `uncertainty_distribution.png` | **Distribuzione delle incertezze** |
| `training_report.pdf` | **Report completo in PDF** |

### Interpretazione

- **Bande di Incertezza**: Zone con bande larghe = alta incertezza
- **Scatter Plot**: Punti vicini alla diagonale = buone predizioni
- **Distribuzione Incertezze**: Mostra dove il modello è più/meno sicuro

---

## 🔧 Configurazione Avanzata

### Modificare l'Architettura del Modello

In `configs/example_config.py`:

```python
'model': {
    'model_type': 'custom',
    'hidden_sizes': [128, 64, 32],  # ← Cambia numero neuroni
    'dropout_rate': 0.2,            # ← Dropout per regolarizzazione
    'use_batchnorm': True,          # ← Batch normalization
}
```

### Modificare il Training

```python
'training': {
    'batch_size': 64,              # ← Batch più grandi = più veloce
    'epochs': 500,                 # ← Più epoche = più training
    'learning_rate': 0.001,        # ← Learning rate più alto = più veloce
    'patience': 50,                # ← Early stopping patience
}
```

---

## 🧪 Testare il Modello Addestrato

Dopo il training, puoi usare il modello per fare predizioni:

```bash
python predict.py
```

O usarlo programmaticamente:

```python
import torch
from src.models import UncertaintyPredictor

# Carica il modello
model = UncertaintyPredictor(...)
model.load_state_dict(torch.load('checkpoints_uncertainty/best_model.pth'))

# Fai predizioni
mean, variance = model(input_data)
```

---

## 📝 Esempi Completi

### Esempio 1: Dati di Produzione

```python
# configs/example_config.py
CONFIG = {
    'data': {
        'csv_path': 'manufacturing.csv',
        'input_columns': ['temp', 'pressure', 'speed', 'material'],
        'output_columns': ['defect_rate'],
    }
}
```

### Esempio 2: Dati Finanziari

```python
CONFIG = {
    'data': {
        'csv_path': 'stock_data.csv',
        'input_columns': ['volume', 'price', 'ma_50', 'rsi'],
        'output_columns': ['next_day_price'],
    }
}
```

### Esempio 3: Dati Medici

```python
CONFIG = {
    'data': {
        'csv_path': 'patient_data.csv',
        'input_columns': ['age', 'bmi', 'blood_pressure', 'glucose'],
        'output_columns': ['risk_score'],
    }
}
```

---

## ❓ Troubleshooting

### Errore: "File not found"

```bash
# Verifica che il CSV sia nella cartella corretta
ls -la uncertainty_predictor/
```

### Errore: "Column not found"

Verifica che i nomi delle colonne in `input_columns` e `output_columns`
corrispondano esattamente ai nomi nel CSV (case-sensitive).

### Training troppo lento

- Riduci `epochs` (es: 200 invece di 400)
- Aumenta `batch_size` (es: 64 invece di 32)
- Usa GPU se disponibile (automatico se installato PyTorch con CUDA)

### Predizioni imprecise

- Aumenta `epochs` (es: 600)
- Prova `model_type: 'large'` invece di 'custom'
- Raccogli più dati di training
- Prova `scaling_method: 'minmax'` invece di 'standard'

---

## 📚 Risorse Aggiuntive

- [README.md](README.md) - Documentazione completa del modulo
- [README_SCM_INTEGRATION.md](../README_SCM_INTEGRATION.md) - Dettagli sul generatore SCM
- [scm/README.md](scm/README.md) - Documentazione generatore dati sintetici

---

## 🎓 Cosa Fa il Modello

Questo modello non predice solo un valore, ma anche l'**incertezza** della predizione:

- **μ(x)**: Valore medio predetto
- **σ²(x)**: Varianza (incertezza) della predizione

**Esempio pratico**:
```
Input: temperatura=25°C, pressione=100kPa
Output:
  - Qualità predetta (μ) = 95.2%
  - Incertezza (σ) = ±2.1%
  - Intervallo 95%: [91.0%, 99.4%]
```

Questo permette di:
✅ Sapere quando il modello è sicuro vs incerto
✅ Prendere decisioni basate sul rischio
✅ Identificare zone di dati problematiche
