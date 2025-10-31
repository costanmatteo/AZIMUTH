# Data Pipeline - Uncertainty Predictor

Guida completa al sistema di gestione dati per il training della rete neurale.

---

## 📋 Indice

1. [Panoramica](#panoramica)
2. [Workflow dei Dati](#workflow-dei-dati)
3. [File e loro Funzioni](#file-e-loro-funzioni)
4. [Configurazione Processi](#configurazione-processi)
5. [Guida all'Uso](#guida-alluso)
6. [Esempi Pratici](#esempi-pratici)

---

## Panoramica

Il sistema gestisce dati di produzione da 5 processi manifatturieri:
- **Laser**
- **Plasma**
- **Galvanic**
- **Multibond**
- **Microetch**

I dati possono essere forniti in formato **Excel** o **CSV**.

---

## Workflow dei Dati

```
┌─────────────────┐
│  File Sorgente  │
│  (.xlsx / .csv) │
└────────┬────────┘
         │
         ├── Se Excel ──────┐
         │                  ▼
         │        ┌──────────────────────┐
         │        │ excel_to_csv_converter│
         │        │ Estrae solo colonne  │
         │        │ rilevanti e rinomina │
         │        └──────────┬───────────┘
         │                   │
         │                   ▼
         │              ┌─────────┐
         └── Se CSV ───→│   CSV   │
                        └────┬────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  preprocessing.py│
                   │  Normalizza dati │
                   │  Split train/val │
                   └────────┬─────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  dataset.py  │
                     │ PyTorch      │
                     │ Dataset      │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   TRAINING   │
                     │ Neural Network│
                     └──────────────┘
```

---

## File e loro Funzioni

### 1. **excel_to_csv_converter.py**
🎯 **Scopo**: Converte file Excel in CSV strutturati

**Cosa fa**:
- Legge file Excel grezzi
- Estrae **SOLO** le colonne definite nella configurazione
- Rinomina colonne con prefissi standard (las_, pla_, gal_, mul_, mic_)
- Parsifica date con il formato corretto per ogni processo
- Salva CSV pulito e strutturato

**Input**: File Excel grezzo con molte colonne
**Output**: CSV con solo le colonne necessarie e nomi standardizzati

**Non fa**: Non normalizza, non scala, non split dei dati

---

### 2. **preprocessing.py**
🎯 **Scopo**: Prepara dati CSV per il training

**Cosa fa**:
- Carica CSV (generato dal converter o già esistente)
- Normalizza/scala i dati (StandardScaler o MinMaxScaler)
- Gestisce valori mancanti
- Split train/validation/test
- Salva gli scaler per l'inferenza

**Input**: File CSV strutturato
**Output**: Numpy arrays normalizzati pronti per il training

**Non fa**: Non legge Excel, non converte formati

---

### 3. **dataset.py**
🎯 **Scopo**: Crea PyTorch Dataset

**Cosa fa**:
- Converte numpy arrays in torch tensors
- Fornisce interfaccia compatibile con PyTorch DataLoader
- Statistiche sui dati

**Input**: Numpy arrays da preprocessing
**Output**: PyTorch Dataset per il training

---

## Configurazione Processi

Ogni processo ha una configurazione specifica che definisce quali colonne estrarre dall'Excel e come rinominarle.

### **Laser**
```python
{
    "process_label": "Laser",          # Colonna processo → las_process
    "hidden_label": "Process_1",       # Colonna processo nascosto → las_hidden_process
    "machine_label": "Machine",        # Colonna macchina → las_machine
    "WA_label": "WA",                  # Colonna work area → las_wa
    "panel_label": "PanelNr",          # Colonna pannello → las_panel
    "PaPos_label": "PaPosNr",          # Colonna posizione → las_papos
    "date_label": ["TimeStamp", "CreateDate 1"],  # Cerca queste colonne per la data
    "date_format": "%m/%d/%y %I:%M %p",           # Formato: 10/31/24 2:30 PM
    "prefix": "las",                              # Prefisso colonne output
    "filename": "laser.csv",                      # Nome file output
    "sep": ",",
    "header": 0
}
```

### **Plasma**
```python
{
    "process_label": "Plasma",
    "hidden_label": "Process_2",
    "machine_label": "Machine",
    "WA_label": "WA",
    "panel_label": "PanelNummer",
    "PaPos_label": "Position",
    "date_label": ["Buchungsdatum"],
    "date_format": "%m/%d/%y %I:%M %p",
    "prefix": "pla",
    "filename": "plasma_fixed.csv",
    "sep": ",",
    "header": 0
}
```

### **Galvanic**
```python
{
    "process_label": "Galvanic",
    "hidden_label": "Process_3",
    "machine_label": null,              # ⚠️ Non presente per questo processo
    "WA_label": "WA",
    "panel_label": "Panelnr",
    "PaPos_label": "PaPosNr",
    "date_label": ["Date/Time Stamp"],
    "date_format": "%m/%d/%y %I:%M %p",
    "prefix": "gal",
    "filename": "galvanik.csv",
    "sep": ",",
    "header": 0
}
```

### **Multibond**
```python
{
    "process_label": "Multibond",
    "hidden_label": "Process_4",
    "machine_label": null,              # ⚠️ Non presente
    "WA_label": "WA",
    "panel_label": null,                # ⚠️ Non presente
    "PaPos_label": "PaPosNr",
    "date_label": ["t_StartDateTime"],
    "date_format": "%m/%d/%y %I:%M %p",
    "prefix": "mul",
    "filename": "multibond.csv",
    "sep": ",",
    "header": 0
}
```

### **Microetch**
```python
{
    "process_label": "Microetch",
    "hidden_label": "Process_5",
    "machine_label": null,              # ⚠️ Non presente
    "WA_label": "WA",
    "panel_label": null,                # ⚠️ Non presente
    "PaPos_label": "PaPosNr",
    "date_label": ["CreateDate"],
    "date_format": "%d.%m.%Y %H:%M:%S", # ⚠️ Formato diverso: 31.10.2024 14:30:00
    "prefix": "mic",
    "filename": "microetch.csv",
    "sep": ",",
    "header": 0
}
```

---

## Guida all'Uso

### Scenario 1: Ho un file Excel grezzo

**Step 1**: Converti Excel → CSV
```python
from data.excel_to_csv_converter import convert_process_file

# Converti file Excel
csv_path = convert_process_file(
    process_name='laser',              # Nome processo
    input_path='/path/to/laser.xlsx',  # File Excel grezzo
    output_dir='./processed_data'      # Dove salvare CSV
)
# Output: ./processed_data/laser.csv (pulito e strutturato)
```

**Step 2**: Preprocessa CSV per training
```python
from data.preprocessing import load_csv_data, DataPreprocessor

# Carica CSV
X, y = load_csv_data(
    filepath='./processed_data/laser.csv',
    input_columns=['las_machine', 'las_wa', 'las_panel'],  # Features
    output_columns=['las_papos']                            # Target
)

# Preprocessa
preprocessor = DataPreprocessor(scaling_method='standard')
X_scaled, y_scaled = preprocessor.fit_transform(X, y)

# Split
X_train, X_val, X_test, y_train, y_val, y_test = \
    preprocessor.split_data(X_scaled, y_scaled)
```

**Step 3**: Crea PyTorch Dataset
```python
from data.dataset import MachineryDataset
from torch.utils.data import DataLoader

# Crea datasets
train_dataset = MachineryDataset(X_train, y_train)
val_dataset = MachineryDataset(X_val, y_val)
test_dataset = MachineryDataset(X_test, y_test)

# Crea DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

### Scenario 2: Ho già un CSV strutturato

**Salta lo Step 1**, vai direttamente a preprocessing:

```python
from data.preprocessing import load_csv_data, DataPreprocessor

# Carica CSV direttamente
X, y = load_csv_data(
    filepath='/path/to/existing_data.csv',
    input_columns=['feature1', 'feature2', 'feature3'],
    output_columns=['target']
)

# Continua con preprocessing come sopra...
```

---

## Esempi Pratici

### Esempio Completo: Training da Excel

```python
# ============================================
# COMPLETO: Da Excel a Neural Network Training
# ============================================

# 1. CONVERSIONE: Excel → CSV
from data.excel_to_csv_converter import convert_process_file

csv_path = convert_process_file(
    process_name='plasma',
    input_path='raw_data/plasma_2024.xlsx',
    output_dir='./processed_data'
)
print(f"CSV creato: {csv_path}")

# 2. PREPROCESSING: CSV → Dati normalizzati
from data.preprocessing import load_csv_data, DataPreprocessor

X, y = load_csv_data(
    filepath=csv_path,
    input_columns=['pla_machine', 'pla_wa', 'pla_panel', 'pla_papos'],
    output_columns=['pla_hidden_process']  # Target da predire
)

preprocessor = DataPreprocessor(scaling_method='standard')
X_scaled, y_scaled = preprocessor.fit_transform(X, y)

# Split dati
X_train, X_val, X_test, y_train, y_val, y_test = \
    preprocessor.split_data(X_scaled, y_scaled,
                           train_size=0.7,
                           val_size=0.15,
                           test_size=0.15)

# Salva scaler per inferenza
preprocessor.save_scalers('./models/plasma_scaler.pkl')

# 3. DATASET: Numpy → PyTorch
from data.dataset import MachineryDataset
from torch.utils.data import DataLoader

train_dataset = MachineryDataset(X_train, y_train)
val_dataset = MachineryDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 4. TRAINING
import torch
import torch.nn as nn

# ... definisci il modello ...

for epoch in range(100):
    for x_batch, y_batch in train_loader:
        # Training step
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        # ... backprop ...
```

---

### Esempio: Conversione multipla

```python
from data.excel_to_csv_converter import convert_all_processes

# Converti tutti i processi contemporaneamente
input_files = {
    'laser': 'raw/laser_data.xlsx',
    'plasma': 'raw/plasma_data.xlsx',
    'galvanic': 'raw/galvanic_data.xlsx',
    'multibond': 'raw/multibond_data.xlsx',
    'microetch': 'raw/microetch_data.xlsx',
}

results = convert_all_processes(input_files, output_dir='./processed_data')

# Verifica risultati
for process, csv_path in results.items():
    if csv_path:
        print(f"✓ {process}: {csv_path}")
    else:
        print(f"✗ {process}: FAILED")
```

---

### Esempio: Verifica colonne Excel

Prima di convertire, puoi verificare quali colonne servono:

```python
from data.excel_to_csv_converter import PROCESS_CONFIGS
import pandas as pd

# Carica Excel per verificare
df = pd.read_excel('raw/laser_data.xlsx')
print("Colonne disponibili nell'Excel:")
print(df.columns.tolist())

print("\nColonne richieste per 'laser':")
config = PROCESS_CONFIGS['laser']
print(f"  - Process: {config['process_label']}")
print(f"  - Hidden: {config['hidden_label']}")
print(f"  - Machine: {config['machine_label']}")
print(f"  - WA: {config['WA_label']}")
print(f"  - Panel: {config['panel_label']}")
print(f"  - PaPos: {config['PaPos_label']}")
print(f"  - Date: {config['date_label']}")
```

---

## Colonne Output CSV

Dopo la conversione, il CSV avrà queste colonne (esempio per Laser):

| Colonna originale Excel | Colonna CSV output | Descrizione |
|------------------------|-------------------|-------------|
| Laser | `las_process` | Nome processo |
| Process_1 | `las_hidden_process` | ID processo nascosto |
| Machine | `las_machine` | Identificativo macchina |
| WA | `las_wa` | Work Area |
| PanelNr | `las_panel` | Numero pannello |
| PaPosNr | `las_papos` | Posizione pannello |
| TimeStamp | `las_timestamp` | Data/ora operazione |

Le colonne **non elencate** nell'Excel verranno **ignorate** durante la conversione.

---

## Note Importanti

### ⚠️ File Excel vs CSV

- **Se hai Excel**: DEVI usare `excel_to_csv_converter.py` prima
- **Se hai CSV**: Vai direttamente a `preprocessing.py`
- **Non mescolare**: Il converter non modifica CSV esistenti

### ⚠️ Nomi Colonne

- I nomi delle colonne nell'Excel **devono corrispondere esattamente** a quelli nella configurazione
- I nomi sono **case-sensitive** (maiuscole/minuscole importano)
- Se una colonna non esiste, verrà saltata (con warning nel log)

### ⚠️ Date

- Ogni processo ha un **formato data diverso**
- Il converter prova a parsificare automaticamente se il formato specificato non funziona
- Date non parsificabili diventano `null` nel CSV

### ⚠️ Colonne Null

- Alcuni processi non hanno tutte le colonne (es. Galvanic non ha `machine_label`)
- Le colonne `null` nella config vengono saltate
- Questo è normale e previsto

---

## Troubleshooting

### "No relevant columns found"
❌ **Problema**: Nessuna colonna della configurazione trovata nell'Excel
✅ **Soluzione**: Verifica che i nomi delle colonne nell'Excel corrispondano alla configurazione

### "process_name must be one of..."
❌ **Problema**: Nome processo non valido
✅ **Soluzione**: Usa uno di: 'laser', 'plasma', 'galvanic', 'multibond', 'microetch'

### Date non parsificate correttamente
❌ **Problema**: Date mostrate come `null` o `NaT`
✅ **Soluzione**: Verifica il formato date nell'Excel e confronta con `date_format` nella config

### File già esistente
❌ **Problema**: Il CSV di output esiste già
✅ **Soluzione**: Il converter sovrascrive automaticamente. Fai backup se necessario.

---

## File di Test

Per testare il converter:
```bash
cd uncertainty_predictor/src/data
python test_converter.py
```

Questo crea file Excel di esempio e verifica che la conversione funzioni.

---

## Domande Frequenti

**Q: Posso modificare le configurazioni dei processi?**
A: Sì, modifica `PROCESS_CONFIGS` in `excel_to_csv_converter.py`

**Q: Posso aggiungere un nuovo processo?**
A: Sì, aggiungi una nuova entry in `PROCESS_CONFIGS` con la stessa struttura

**Q: Il converter modifica l'Excel originale?**
A: No, l'Excel originale non viene mai modificato

**Q: Dove vengono salvati i CSV?**
A: Nella directory specificata in `output_dir` (default: `./output`)

**Q: Posso usare lo stesso CSV per più esperimenti?**
A: Sì, il CSV è indipendente. Puoi riusarlo con preprocessing diversi

---

## Contatti e Supporto

Per problemi o domande, consulta:
- `README_converter.md` - Documentazione dettagliata del converter
- `convert_example.py` - Esempi di utilizzo
- `test_converter.py` - Test di verifica

---

**Versione**: 1.0
**Ultimo aggiornamento**: 2024-10-31
