# Controller Optimization

Sistema per ottimizzare sequenze di processi manufacturing attraverso policy generators.

## Overview

Il sistema è composto da:
- **Uncertainty Predictors**: Modelli che predicono output + varianza per ogni processo (frozen)
- **Policy Generators**: Reti neurali che generano parametri ottimali per il processo successivo (trainable)
- **Surrogate Model**: Valutazione della reliability della trajectory completa

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
```

### Principio Fondamentale

**NON creare classi diverse per ogni processo!**

Invece:
- **UNA SOLA classe** `UncertaintyPredictor` (già esistente in `uncertainty_predictor/`)
- **UNA SOLA classe** `PolicyGenerator` (in questo modulo)
- **Pesi diversi salvati su disco** per ogni processo

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
1. Addestra un uncertainty predictor per ogni processo (laser, plasma, galvanic)
2. Genera dataset SCM con physical noise models
3. Salva modelli e preprocessor in `checkpoints/{process}/`
4. **Genera report PDF identico a quello di `uncertainty_predictor`** per ogni processo

Output:
- `checkpoints/laser/uncertainty_predictor.pth`
- `checkpoints/laser/scalers.pkl`
- `checkpoints/laser/training_info.json`
- `checkpoints/laser/training_report.pdf` ← **Report completo con metriche**
- `checkpoints/laser/*.png` (plots)

E lo stesso per `plasma/` e `galvanic/`.

**Report per Processo**

Ogni processo genera un report PDF identico a quello di `uncertainty_predictor`, contenente:
- Configurazione modello e dataset
- Metriche test (MSE, RMSE, MAE, R², Calibration Ratio, NLL)
- Training history plots
- Predictions with uncertainty bounds
- Scatter plots with uncertainty coloring
- SCM graph

### Step 2: Train Controller (Policy Generators)

```bash
python train_controller.py
```

Questo comando:
1. Carica gli uncertainty predictors addestrati (frozen)
2. Crea policy generators (trainable)
3. Genera target trajectory **a\*** con noise=0 (ottimale)
4. Genera baseline trajectory **a'** con noise normale (NO controller)
5. Addestra i policy generators per minimizzare `L = (F - F*)² + λ_BC * Σ||a_t - a_t*||²`
6. Valuta la trajectory finale **a** con controller
7. **Genera report finale con confronto completo**

Output:
- `checkpoints/controller/policy_0.pth`, `policy_1.pth` (policy generators)
- `checkpoints/controller/training_history.json`
- `checkpoints/controller/final_results.json`
- `checkpoints/controller/controller_report.pdf` ← **Report sistema completo**
- `checkpoints/controller/*.png` (comparison plots)

## Report Generati

### Report Controller (Step 2)

Il report finale del sistema completo contiene:

#### 1. Configuration
- Processi utilizzati
- Architettura policy generators
- Parametri training (epochs, learning rate, λ_BC, etc.)

#### 2. Final Metrics - Tabella Comparativa

| Metric | Value |
|--------|-------|
| Target Reliability (F*) | 0.9500 |
| Baseline Reliability (F') | 0.8200 |
| **Controller Reliability (F)** | **0.9350** |
| **Improvement over Baseline** | **+14.0%** |
| Gap from Target | 1.6% |

#### 3. Process-wise Metrics

| Process | Input MSE | Output MSE | Combined MSE |
|---------|-----------|------------|--------------|
| Laser | 0.0012 | 0.0008 | 0.0010 |
| Plasma | 0.0034 | 0.0021 | 0.0028 |
| Galvanic | 0.0056 | 0.0042 | 0.0049 |

#### 4. Visualizations
- **Training History**: Evolution of total loss, reliability loss, BC loss, F values
- **Trajectory Comparison**: Side-by-side comparison of a*, a', a for each process
- **Reliability Bar Chart**: Visual comparison of F*, F', F
- **Process Improvements**: MSE reduction per process

## Struttura Directory

```
controller_optimization/
├── src/
│   ├── models/
│   │   ├── policy_generator.py         # PolicyGenerator model
│   │   └── surrogate.py                # ProT surrogate (placeholder)
│   ├── training/
│   │   ├── process_trainer.py          # Train uncertainty predictors
│   │   └── controller_trainer.py       # Train policy generators
│   └── utils/
│       ├── model_utils.py              # Load/save helpers
│       ├── target_generation.py        # Generate a* and a'
│       ├── process_chain.py            # Orchestration
│       ├── metrics.py                  # Evaluation metrics
│       ├── visualization.py            # Plots
│       └── report_generator.py         # PDF reports
│
├── configs/
│   ├── processes_config.py             # Process definitions
│   └── controller_config.py            # Controller training config
│
├── checkpoints/                        # Generated during training
│   ├── laser/
│   │   ├── uncertainty_predictor.pth
│   │   ├── scalers.pkl
│   │   ├── training_info.json
│   │   ├── training_report.pdf         # Report processo
│   │   └── *.png
│   ├── plasma/ (same structure)
│   ├── galvanic/ (same structure)
│   └── controller/
│       ├── policy_0.pth, policy_1.pth
│       ├── training_history.json
│       ├── final_results.json
│       ├── controller_report.pdf       # Report sistema completo
│       └── *.png
│
├── train_processes.py                  # Step 1: Train predictors
├── train_controller.py                 # Step 2: Train controller
├── requirements.txt
└── README.md
```

## Metriche nel Report Controller

### Reliability Scores

- **F\* (Target)**: Reliability della trajectory ottimale (noise=0)
- **F' (Baseline)**: Reliability senza controller (noise normale)
- **F (Controller)**: Reliability con policy generators

### Obiettivo

Vogliamo dimostrare che: **F ≈ F\*** e **F > F'**

Ovvero, il controller riesce a:
1. Avvicinarsi alla performance ottimale (gap da F* piccolo)
2. Migliorare significativamente rispetto al baseline senza controllo

### Improvement Metric

```
Improvement = (F - F') / F' × 100%
```

Esempio: Se F'=0.82 e F=0.935, allora Improvement = +14.0%

## Advanced Usage

### Train Specific Processes Only

```bash
python train_processes.py --processes laser plasma
```

### Skip Already Trained Processes

```bash
python train_processes.py --skip-existing
```

### Use GPU

```bash
python train_processes.py --device cuda
python train_controller.py  # Auto-detects GPU
```

### Custom Configuration

Modifica `configs/controller_config.py`:

```python
CONTROLLER_CONFIG = {
    'policy_generator': {
        'architecture': 'large',  # or 'small', 'medium', 'custom'
        'hidden_sizes': [128, 64, 32],  # se 'custom'
        'dropout': 0.15,
    },
    'training': {
        'epochs': 200,
        'learning_rate': 0.0005,
        'lambda_bc': 0.2,  # Higher BC weight
    },
}
```

## Extending the System

### Add a New Process

1. Definisci il processo in `configs/processes_config.py`:

```python
{
    'name': 'microetch',
    'scm_dataset_type': 'microetch',
    'input_dim': 3,
    'output_dim': 1,
    'input_labels': ['Temperature', 'Concentration', 'Duration'],
    'output_labels': ['RemovalDepth'],
    'uncertainty_predictor': { ... },
    'checkpoint_dir': 'checkpoints/microetch',
}
```

2. Aggiungi alla lista `PROCESSES`

3. Ri-esegui:
```bash
python train_processes.py --processes microetch
python train_controller.py  # Auto-include nuovo processo
```

## Troubleshooting

### Error: "Uncertainty predictor not found"

**Soluzione**: Esegui prima `train_processes.py` per addestrare gli uncertainty predictors.

### Error: "No trainable parameters found"

**Causa**: Gli uncertainty predictors devono essere frozen e i policy generators trainable.

**Verifica**: Controlla che ProcessChain carichi correttamente i modelli.

### Training diverges (F decreases)

**Possibili soluzioni**:
1. Riduci learning rate: `learning_rate: 0.0005`
2. Aumenta BC weight: `lambda_bc: 0.2`
3. Usa architettura più piccola: `architecture: 'small'`

## References

- **PDF Tecnico**: `Azimuth.pdf` (architecture specification)
- **Uncertainty Predictor Module**: `../uncertainty_predictor/`
- **SCM Datasets**: `../uncertainty_predictor/scm_ds/datasets.py`

## Citation

Se usi questo sistema, cita:

```
AZIMUTH Controller Optimization System
Università degli Studi di Milano
2025
```
