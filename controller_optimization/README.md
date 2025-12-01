# Controller Optimization

Sistema per ottimizzare sequenze di processi manufacturing attraverso policy generators con uncertainty quantification.

## Overview

Il sistema ottimizza una catena di 4 processi manufacturing:
1. **Laser Drilling** - Foratura laser
2. **Plasma Cleaning** - Pulizia al plasma
3. **Galvanic Deposition** - Deposizione galvanica
4. **Micro-Etching** - Micro-incisione

### Componenti Principali

| Componente | Descrizione | Stato durante training |
|------------|-------------|------------------------|
| **Uncertainty Predictors** | Predicono output + varianza per ogni processo | Frozen |
| **Policy Generators** | Generano parametri ottimali per il processo successivo | Trainable |
| **Surrogate Model** | Valuta la reliability della trajectory completa | Fixed |

---

## Architettura

```
Processo 1 (Laser):
  a1 (fisso) → UncertPred1 → (o1, σ1²)
                    ↓
Processo 2 (Plasma):
  [a1, o1, σ1²] → Policy1 → a2 → UncertPred2 → (o2, σ2²)
                                        ↓
Processo 3 (Galvanic):
  [a2, o2, σ2²] → Policy2 → a3 → UncertPred3 → (o3, σ3²)
                                        ↓
Processo 4 (Microetch):
  [a3, o3, σ3²] → Policy3 → a4 → UncertPred4 → (o4, σ4²)
                                        ↓
                              Surrogate → F (reliability)
```

### Principio Fondamentale

**NON creare classi diverse per ogni processo!**

Invece:
- **UNA SOLA classe** `UncertaintyPredictor` (in `uncertainty_predictor/`)
- **UNA SOLA classe** `PolicyGenerator` (in questo modulo)
- **Pesi diversi salvati su disco** per ogni processo

---

## Quick Start

### Prerequisiti

```bash
cd controller_optimization
pip install -r requirements.txt
```

### Step 1: Train Uncertainty Predictors

```bash
python train_processes.py
```

Questo comando:
1. Addestra un uncertainty predictor per ogni processo
2. Genera dataset SCM con physical noise models
3. Salva modelli e preprocessor in `checkpoints/{process}/`
4. Genera report PDF per ogni processo

**Output:**
```
checkpoints/
├── laser/
│   ├── uncertainty_predictor.pth    # Modello
│   ├── scalers.pkl                  # Preprocessori
│   ├── training_info.json           # Metriche
│   └── training_report.pdf          # Report completo
├── plasma/
├── galvanic/
└── microetch/
```

### Step 2: Train Controller (Policy Generators)

```bash
python train_controller.py
```

Questo comando:
1. Carica gli uncertainty predictors addestrati (frozen)
2. Crea policy generators (trainable)
3. Genera **50 scenari diversi** con condizioni ambientali variabili
4. Addestra i policy generators per minimizzare la reliability loss
5. Genera report finale con analisi multi-scenario

**Output:**
```
checkpoints/controller/
├── policy_0.pth, policy_1.pth, ...  # Policy generators
├── training_history.json             # Storico training
├── final_results.json                # Risultati aggregati
├── controller_report.pdf             # Report completo
├── theoretical_analysis_data.json    # Analisi teorica
└── *.png                             # Visualizzazioni
```

---

## Multi-Scenario Training

Il controller addestra su **50 scenari diversi** invece di un singolo punto operativo.

### Come Funziona

1. **Generazione Target (a\*)**: 50 scenari con condizioni strutturali diverse
   - Laser: AmbientTemp varia tra 15-35°C
   - Microetch: Temperature varia tra 293-323K
   - Process noise = 0 (comportamento ideale)

2. **Generazione Baseline (a')**: Stesse condizioni strutturali
   - Process noise attivo (comportamento realistico)
   - Confronto equo: stesse condizioni ambientali

3. **Training**: Il controller apprende ad adattarsi a tutte le condizioni

4. **Valutazione**: Metriche aggregate su tutti gli scenari

### Metriche di Output

```
FINAL RESULTS - AGGREGATED OVER ALL SCENARIOS
============================================
Number of scenarios:           50

F* (target, optimal):
  Mean:  0.849 ± 0.003
  Range: [0.843, 0.855]

F' (baseline, no controller):
  Mean:  0.821 ± 0.005
  Range: [0.810, 0.832]

F  (actual, with controller):
  Mean:  0.846 ± 0.004
  Range: [0.838, 0.854]

Improvement over baseline:     +3.05%
Gap from optimal:              0.35%
Robustness (std of F):         0.004  (lower = more robust)
============================================
```

---

## Report Generati

### Report Processo (Step 1)

Ogni processo genera un report PDF contenente:
- Configurazione modello e dataset
- Metriche test (MSE, RMSE, MAE, R², Calibration Ratio, NLL)
- Training history plots
- Predictions with uncertainty bounds
- Scatter plots with uncertainty coloring

### Report Controller (Step 2)

Il report finale contiene:

#### 1. Configuration
- Processi utilizzati
- Architettura policy generators
- Parametri training

#### 2. Final Metrics - Multi-Scenario

| Metric | Mean | Std | Range |
|--------|------|-----|-------|
| F* (Target) | 0.849 | 0.003 | [0.843, 0.855] |
| F' (Baseline) | 0.821 | 0.005 | [0.810, 0.832] |
| F (Controller) | 0.846 | 0.004 | [0.838, 0.854] |

#### 3. Process-wise Metrics

| Process | Input MSE | Output MSE |
|---------|-----------|------------|
| Laser | 0.0012 | 0.0008 |
| Plasma | 0.0034 | 0.0021 |
| Galvanic | 0.0056 | 0.0042 |
| Microetch | 0.0067 | 0.0051 |

#### 4. Visualizations
- Training history
- Trajectory comparison (a*, a', a)
- Reliability bar chart
- Per-scenario analysis

---

## Struttura Directory

```
controller_optimization/
├── src/
│   ├── models/
│   │   ├── policy_generator.py        # PolicyGenerator network
│   │   ├── surrogate.py               # ProT surrogate model
│   │   └── scenario_encoder.py        # Scenario encoding
│   │
│   ├── training/
│   │   ├── process_trainer.py         # Train uncertainty predictors
│   │   └── controller_trainer.py      # Train policy generators
│   │
│   ├── analysis/
│   │   ├── theoretical_loss_analysis.py    # L_min computation
│   │   ├── theoretical_visualization.py    # Plots
│   │   ├── theoretical_tables.py           # Tables
│   │   └── THEORETICAL_LOSS_ANALYSIS.md    # Documentation
│   │
│   └── utils/
│       ├── model_utils.py             # Load/save helpers
│       ├── target_generation.py       # Generate a* and a'
│       ├── process_chain.py           # Orchestration
│       ├── metrics.py                 # Evaluation metrics
│       ├── visualization.py           # Plots
│       └── report_generator.py        # PDF reports
│
├── configs/
│   ├── processes_config.py            # Process definitions
│   └── controller_config.py           # Controller training config
│
├── checkpoints/                       # Generated during training
├── train_processes.py                 # Step 1
├── train_controller.py                # Step 2
├── requirements.txt
└── README.md
```

---

## Loss Function

Il controller minimizza:

```
L = scale × (F - F*)² + λ_BC × Σ||a_t - a_t*||²
```

Dove:
- **F**: Reliability della trajectory con controller
- **F\***: Reliability della trajectory target (ottimale)
- **λ_BC**: Peso del behavior cloning loss
- **scale**: Fattore di scala (default: 100.0)

### Reliability Scores

- **F\* (Target)**: Reliability ottimale (noise=0)
- **F' (Baseline)**: Reliability senza controller (noise normale)
- **F (Controller)**: Reliability con policy generators

### Obiettivo

Vogliamo: **F ≈ F\*** e **F > F'**

---

## Analisi Teorica

Il modulo include un framework per calcolare la **loss minima raggiungibile (L_min)**.

### Concetti Chiave

- **L_min**: Loss irriducibile dovuta alla stocasticità del sampling
- **Var[F]**: Componente di varianza (irriducibile)
- **Bias²**: Bias sistematico (riducibile con training)
- **Efficiency**: L_min / loss osservata (100% = ottimale)

### Formula

```
E[L] = Var[F] + Bias²

L_min = Var[F]  (quando Bias → 0)
```

Quando σ² > 0, abbiamo sempre **L_min > 0** perché il sampling stocastico introduce varianza irriducibile.

Vedi [THEORETICAL_LOSS_ANALYSIS.md](src/analysis/THEORETICAL_LOSS_ANALYSIS.md) per dettagli matematici.

---

## Configurazione

### Configurazione Processi

`configs/processes_config.py`:

```python
PROCESSES = [
    {
        'name': 'laser',
        'scm_dataset_type': 'laser',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'output_labels': ['ActualPower'],
        'controllable_inputs': [0],  # Solo PowerTarget
        'structural_inputs': [1],    # AmbientTemp è ambientale
        'uncertainty_predictor': {
            'hidden_sizes': [64, 32, 16],
            'dropout_rate': 0.1,
        },
    },
    # ... altri processi
]
```

### Configurazione Controller

`configs/controller_config.py`:

```python
CONTROLLER_CONFIG = {
    'target': {
        'n_samples': 50,      # Numero di scenari
        'seed': 42
    },
    'baseline': {
        'n_samples': 50,      # Deve coincidere con target
        'seed': 43
    },
    'policy_generator': {
        'architecture': 'medium',  # 'small', 'medium', 'large', 'custom'
        'hidden_sizes': [64, 32, 16],
        'dropout': 0.1,
    },
    'training': {
        'epochs': 200,
        'learning_rate': 0.001,
        'lambda_bc': 0.1,
        'reliability_loss_scale': 100.0,
    },
}
```

---

## Advanced Usage

### Train Solo Specifici Processi

```bash
python train_processes.py --processes laser plasma
```

### Salta Processi Già Addestrati

```bash
python train_processes.py --skip-existing
```

### Usa GPU

```bash
python train_processes.py --device cuda
python train_controller.py  # Auto-detect GPU
```

### Test Rapido (5 scenari)

Modifica `controller_config.py`:
```python
'target': {'n_samples': 5, 'seed': 42},
'baseline': {'n_samples': 5, 'seed': 43},
```

---

## Classificazione Noise

Il sistema distingue tra due tipi di rumore:

### Structural Noise (Ambientale)
- **AmbientTemp** nel laser (15-35°C)
- **Temperature** nel microetch (293-323K)
- Crea diversità tra scenari
- ATTIVO sia in target che baseline

### Process Noise (Misurazione)
- **Zln, NoiseShot, NoiseMeas, NoiseDrift** nel laser
- **Zln, NoiseAdd, Jump** nel plasma
- **SpatialVar, TimeRand, PhaseRand, NoiseMeas** nel galvanico
- **Zln, NoiseStudentT** nel microetch
- ZERO in target, ATTIVO in baseline

Vedi [NOISE_CLASSIFICATION_ANALYSIS.md](NOISE_CLASSIFICATION_ANALYSIS.md) per dettagli.

---

## Extending the System

### Aggiungere un Nuovo Processo

1. Definisci il processo in `configs/processes_config.py`:

```python
{
    'name': 'new_process',
    'scm_dataset_type': 'new_process',
    'input_dim': 3,
    'output_dim': 1,
    'input_labels': ['Input1', 'Input2', 'Input3'],
    'output_labels': ['Output1'],
    'controllable_inputs': [0, 1],
    'structural_inputs': [2],
    'uncertainty_predictor': { ... },
    'checkpoint_dir': 'checkpoints/new_process',
}
```

2. Aggiungi alla lista `PROCESSES`

3. Se necessario, crea un nuovo SCM dataset in `uncertainty_predictor/scm_ds/datasets.py`

4. Ri-esegui:
```bash
python train_processes.py --processes new_process
python train_controller.py
```

---

## Troubleshooting

### Error: "Uncertainty predictor not found"

**Soluzione**: Esegui prima `train_processes.py`.

### Training diverges (F decreases)

**Soluzioni**:
1. Riduci learning rate: `0.0005`
2. Aumenta BC weight: `0.2`
3. Usa architettura più piccola: `'small'`

### Training lento con 50 scenari

Inizia con 5-10 scenari per test. Usa 50 per produzione.

### L_min = 0 nell'analisi teorica

**Causa**: Il calcolo di delta usa valori errati.

**Verifica**: Controlla che delta = μ_target - τ sia calcolato correttamente dalla target_trajectory.

### Violations nella theoretical analysis

**Possibili cause**:
1. Problemi numerici
2. Distribuzione non-Gaussiana
3. Bias da campioni finiti

---

## Dipendenze

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
reportlab>=4.0.0
pypdf==6.2.0
```

---

## References

- **PDF Tecnico**: `Azimuth.pdf` (architecture specification)
- **Uncertainty Predictor Module**: `../uncertainty_predictor/`
- **SCM Datasets**: `../uncertainty_predictor/scm_ds/datasets.py`
- **Theoretical Analysis**: `src/analysis/THEORETICAL_LOSS_ANALYSIS.md`
- **Noise Classification**: `NOISE_CLASSIFICATION_ANALYSIS.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`

---

## Citation

```
AZIMUTH Controller Optimization System
Università degli Studi di Milano
2025
```
