# Guida all'uso di SCM (Structural Causal Model)

## Cosa è SCM?

SCM (Structural Causal Model) è un sistema per generare **dataset sintetici** con relazioni causali definite matematicamente. Utile per:
- 🧪 Testing e sviluppo senza dati reali
- 📊 Generare grandi quantità di dati
- 🔬 Studiare relazioni causali note

## Come usare SCM

### Opzione 1: Usare SCM (dataset sintetico)

Nel file `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': None,  # ← Imposta a None per usare SCM
        'use_scm': True,   # ← Abilita SCM
        'scm': {
            'n_samples': 5000,         # Numero di campioni da generare
            'seed': 42,                # Seed per riproducibilità
            'dataset_type': 'one_to_one_ct'  # Tipo di SCM
        },
        # ... resto configurazione
    }
}
```

Poi esegui:
```bash
python train.py
```

### Opzione 2: Usare un file CSV

Nel file `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'path/to/your/data.csv',  # ← Path al tuo CSV
        'input_columns': ['x', 'y', 'z'],     # Colonne input
        'output_columns': ['res_1'],           # Colonne output
        # ... resto configurazione
    }
}
```

## Parametri SCM disponibili

### `n_samples`
Numero di campioni da generare.
- **Default**: 5000
- **Esempio**: `'n_samples': 10000` per 10k campioni

### `seed`
Seed per la generazione casuale (per riproducibilità).
- **Default**: 42
- **Esempio**: `'seed': 123` per un seed diverso

### `dataset_type`
Tipo di relazione causale nel dataset.
- **Default**: `'one_to_one_ct'`
- **Disponibili**:
  - `'one_to_one_ct'`: Ogni parent ha un child, con cross-talk tra children

## Dataset SCM corrente: 'one_to_one_ct'

Questo dataset genera:
- **10 input features**: P1, P2, P3, P4, P5, C1, C2, C3, C4, C5
- **1 output**: Y

### Struttura causale:
```
P1 → C1 ↘
P2 → C2 ↘
P3 → C3 → Y
P4 → C4 ↗
P5 → C5 ↗
```

Dove:
- **P1-P5**: Parents (variabili esogene indipendenti)
- **C1-C5**: Children (dipendono dai parents)
- **Y**: Output finale (somma di tutti i children + rumore)

### Equazioni strutturali:
```
P1, P2, P3, P4, P5 ~ N(0, 1)  # Noise normale standard
C1 = P1 + ε_C1
C2 = P2 + ε_C2
C3 = P3 + ε_C3
C4 = P4 + ε_C4
C5 = P5 + ε_C5
Y = C1 + C2 + C3 + C4 + C5 + ε_Y
```

## Esempio pratico completo

### 1. Genera 10,000 campioni con SCM

```python
# configs/example_config.py
CONFIG = {
    'data': {
        'csv_path': None,
        'use_scm': True,
        'scm': {
            'n_samples': 10000,  # ← 10k campioni
            'seed': 42,
            'dataset_type': 'one_to_one_ct'
        },
        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    },
    'model': {
        'model_type': 'medium',  # ← Modello medio
        # ...
    },
    'training': {
        'epochs': 200,
        'batch_size': 64,
        # ...
    }
}
```

### 2. Esegui training

```bash
cd uncertainty_predictor
python train.py
```

### 3. Output atteso

```
======================================================================
UNCERTAINTY QUANTIFICATION TRAINING
======================================================================

[1/7] Loading data...
CSV file not specified or not found. Using SCM synthetic data generation...
Generating 10000 synthetic samples using SCM...
SCM data generated: 10000 samples
Input features: 10 - ['P1', 'P2', 'P3', 'P4', 'P5', 'C1', 'C2', 'C3', 'C4', 'C5']
Output features: 1 - ['Y']
  Loaded 10000 samples

[2/7] Preprocessing data...
  Train set: 7000 samples
  Validation set: 1500 samples
  Test set: 1500 samples

[3/7] Creating PyTorch datasets...
  Input dimension: 10
  Output dimension: 1

[4/7] Creating uncertainty model...
  Model type: medium
  Total parameters: 12345

[5/7] Setting up Gaussian NLL Loss...

[6/7] Starting training...
...
```

## Testare SCM manualmente

Se vuoi solo generare dati SCM senza trainare:

```python
# test_scm.py
import sys
from pathlib import Path

sys.path.insert(0, 'uncertainty_predictor/scm_ds')
from datasets import ds_scm_1_to_1_ct

# Genera 100 campioni
df = ds_scm_1_to_1_ct.sample(n=100, seed=42)

print(df.head())
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

## Passare da SCM a CSV e viceversa

### SCM → CSV
Basta cambiare `csv_path` da `None` a un path valido:
```python
'csv_path': 'data/my_real_data.csv',
```

### CSV → SCM
Basta impostare `csv_path` a `None`:
```python
'csv_path': None,
```

Il sistema seleziona automaticamente la sorgente corretta!

## Creare nuovi dataset SCM

Per creare un nuovo tipo di dataset SCM, modifica `scm_ds/datasets.py`:

```python
ds_my_custom = SCMDataset(
    name="my_custom_dataset",
    description="Descrizione del mio dataset",
    tags=None,
    specs=[
        NodeSpec("X", [], "eps_X"),           # Nodo esogeno
        NodeSpec("Y", ["X"], "2*X + eps_Y"),  # Y dipende da X
    ],
    params={},
    singles={
        "X": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.standard_normal(n),
    },
    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)
```

Poi usalo con:
```python
'dataset_type': 'my_custom'
```

## FAQ

**Q: Quale è meglio: CSV o SCM?**
- **CSV**: Usa per dati reali di produzione
- **SCM**: Usa per testing, sviluppo, o studio di relazioni causali note

**Q: Posso cambiare il numero di features in SCM?**
- Sì, ma devi creare un nuovo dataset SCM in `datasets.py`

**Q: I risultati con SCM sono validi?**
- Sì per testing e sviluppo
- Per produzione, usa sempre dati reali

**Q: Posso salvare i dati SCM generati?**
- Sì! Genera i dati e salvali come CSV:
```python
from scm_ds.datasets import ds_scm_1_to_1_ct
df = ds_scm_1_to_1_ct.sample(n=5000, seed=42)
df.to_csv('generated_data.csv', index=False)
```

## Risoluzione problemi

### Errore: "No module named 'sympy'"
```bash
pip install sympy graphviz
```

### Errore: "failed to execute dot"
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

### Il training usa il CSV invece di SCM
Verifica che:
1. `csv_path: None` (non un path)
2. `use_scm: True`
3. Il file CSV non esista al path specificato
