# SCM Surrogate Guide

## Overview

Il surrogato SCM è un'alternativa deterministica agli uncertainty predictors neurali. Invece di usare reti neurali trained per predire gli output dei macchinari, il surrogato usa direttamente le **funzioni strutturali causali (SCM)** definite nel dataset.

## Vantaggi

1. **Deterministico**: Le funzioni SCM sono equazioni matematiche esatte, non approssimazioni neurali
2. **Interpretabile**: Ogni equazione è esplicita e comprensibile
3. **Perfettamente allineato**: Usa le stesse funzioni usate per generare i dati
4. **Differenziabile**: Il gradiente può fluire attraverso le funzioni PyTorch
5. **No training necessario**: Non richiede uncertainty predictors pre-trained

## Come Funziona

### Architettura

```
Input → SCM Surrogate → (Mean, Variance)
```

Il surrogato:
1. Prende gli input del processo (es. PowerTarget, AmbientTemp per laser)
2. Valuta le equazioni SCM in ordine topologico
3. Ritorna:
   - **mean**: Output deterministico dalle funzioni SCM (con rumore = 0)
   - **variance**: Costante piccola (1e-6) per compatibilità con l'interface

### Implementazione

Il surrogato è implementato in:
- `controller_optimization/src/models/scm_surrogate.py`

Caratteristiche principali:
- **SymPy → PyTorch**: Le espressioni SymPy vengono convertite in operazioni PyTorch differenziabili
- **Topological Order**: I nodi vengono valutati nell'ordine causale corretto
- **Zero Noise**: Il rumore esogeno è settato a zero per ottenere funzioni deterministiche

## Utilizzo

### 1. Training del Controller con Surrogato SCM (CONFIGURABILE)

Il modo più semplice è modificare il **config file**:

```python
# controller_optimization/configs/controller_config.py

CONTROLLER_CONFIG = {
    # ... altre configurazioni ...

    'surrogate': {
        'use_scm_surrogate': True,  # <-- Cambia a True per usare surrogato SCM
        'use_deterministic_sampling': True,
    },
}
```

Poi esegui normalmente:
```bash
python controller_optimization/train_controller.py
```

### 1b. Alternativa: Usa Parametro Diretto

Se preferisci, puoi anche passare il parametro direttamente:

```python
# train_controller.py (o tuo script personalizzato)
chain = ProcessChain(
    processes_config=PROCESSES,
    target_trajectory=target_traj,
    policy_config=policy_config,
    device=device,
    use_scm_surrogate=True  # <-- Abilita surrogato SCM
)
```

### 2. Confronto: Uncertainty Predictor vs Surrogato

| Aspetto | Uncertainty Predictor | SCM Surrogate |
|---------|----------------------|---------------|
| **Tipo** | Rete neurale | Funzioni deterministiche |
| **Training** | Richiesto (train_processes.py) | Non necessario |
| **Preprocessing** | Usa StandardScaler | Non necessario (lavora in spazio originale) |
| **Output variance** | Predetta dalla rete | Costante piccola |
| **Interpretabilità** | Blackbox | Equazioni esplicite |
| **Accuratezza** | Dipende dal training | Perfetta (funzioni esatte) |

### 3. Esempio Completo

```python
from controller_optimization.configs.processes_config import PROCESSES
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.utils.target_generation import generate_target_trajectory

# 1. Genera target trajectory
target_traj = generate_target_trajectory(
    processes_config=PROCESSES,
    n_samples=100,
    seed=42
)

# 2. Crea ProcessChain con surrogato SCM
chain = ProcessChain(
    processes_config=PROCESSES,
    target_trajectory=target_traj,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_scm_surrogate=True  # Usa funzioni SCM invece di reti neurali
)

# 3. Forward pass
trajectory = chain.forward(batch_size=32, scenario_idx=0)

# 4. Il gradiente può fluire!
loss = compute_loss(trajectory)
loss.backward()  # ✓ Il gradiente fluisce attraverso le funzioni SCM
```

## Funzioni SCM Implementate

Il surrogato supporta i seguenti processi (definiti in `uncertainty_predictor/scm_ds/datasets.py`):

### 1. Laser Drilling (`ds_scm_laser`)
- **Inputs**: PowerTarget, AmbientTemp
- **Outputs**: ActualPower
- **Modello**: Light-Current-Temperature (L-I-T) con rumore fisico

### 2. Plasma Cleaning (`ds_scm_plasma`)
- **Inputs**: RF_Power, Duration
- **Outputs**: RemovalRate
- **Modello**: Decadimento esponenziale con rumore lognormal

### 3. Galvanic Copper Deposition (`ds_scm_galvanic`)
- **Inputs**: CurrentDensity, Duration
- **Outputs**: Thickness
- **Modello**: Legge di Faraday con variazione spaziale

### 4. Micro-Etching (`ds_scm_microetch`)
- **Inputs**: Temperature, Concentration, Duration
- **Outputs**: RemovalDepth
- **Modello**: Arrhenius kinetics con rumore Student-t

## Dettagli Tecnici

### Conversione SymPy → PyTorch

Il surrogato converte automaticamente le espressioni SymPy in operazioni PyTorch:

| SymPy | PyTorch |
|-------|---------|
| `sp.exp(x)` | `torch.exp(x)` |
| `sp.sqrt(x)` | `torch.sqrt(x)` |
| `sp.log(x)` | `torch.log(x)` |
| `sp.sin(x)` | `torch.sin(x)` |
| `sp.Abs(x)` | `torch.abs(x)` |
| `x ** 2` | `x * x` |

### Gestione del Rumore

Nel dataset SCM, ogni nodo ha un termine di rumore `eps_<nome>`:
- Nel training dell'uncertainty predictor: rumore campionato da distribuzioni specifiche
- Nel surrogato: **rumore = 0** per ottenere funzioni deterministiche

### Differenziabilità

Tutte le operazioni sono implementate usando PyTorch nativo, garantendo:
- ✓ Gradiente computato automaticamente
- ✓ Supporto per GPU
- ✓ Compatibilità con autograd

## Debugging e Verifica

### Test del Surrogato

Esegui il test standalone:
```bash
python controller_optimization/src/models/scm_surrogate.py
```

Verifica:
1. Forward pass funziona correttamente
2. Output shape è corretto
3. Gradiente esiste e ha norma > 0

### Confronto Output

Puoi confrontare gli output del surrogato con quelli del dataset:

```python
# 1. Output dal surrogato
surrogate = create_scm_surrogate_for_process(laser_config)
inputs = torch.tensor([[0.5, 25.0]])  # PowerTarget, AmbientTemp
mean, var = surrogate(inputs)

# 2. Output dal dataset SCM
from uncertainty_predictor.scm_ds.datasets import ds_scm_laser
df = ds_scm_laser.sample(n=1, seed=42)

# 3. Confronta
print(f"Surrogate: {mean.item():.4f}")
print(f"Dataset:   {df['ActualPower'].values[0]:.4f}")
# Nota: piccola differenza dovuta al rumore nel dataset
```

## Limitazioni

1. **Solo per processi nel dataset**: Il surrogato supporta solo i 4 processi definiti in `datasets.py`
2. **Funzioni deterministiche**: Non modella l'incertezza aleatorica (variance sempre piccola)
3. **Conversione manuale**: Alcune funzioni SymPy complesse potrebbero richiedere implementazione manuale

## Next Steps

### Estendere il Surrogato

Per aggiungere un nuovo processo:

1. Definisci le NodeSpec in `uncertainty_predictor/scm_ds/datasets.py`
2. Crea un nuovo `ds_scm_<processo>` dataset
3. Aggiungi il mapping in `scm_surrogate.py`:
   ```python
   dataset_map = {
       'laser': ds_scm_laser,
       'plasma': ds_scm_plasma,
       'galvanic': ds_scm_galvanic,
       'microetch': ds_scm_microetch,
       'nuovo_processo': ds_scm_nuovo_processo  # <-- Aggiungi qui
   }
   ```

### Miglioramenti Futuri

- [ ] Supporto per incertezza parametrica nelle funzioni SCM
- [ ] Caching delle espressioni compilate per performance
- [ ] Validazione automatica output vs dataset
- [ ] Visualizzazione delle funzioni SCM

## References

- **SCM Theory**: Pearl, J. (2009). "Causality: Models, Reasoning and Inference"
- **SymPy Documentation**: https://docs.sympy.org/
- **PyTorch Autograd**: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

## Supporto

Per problemi o domande:
1. Verifica che PyTorch sia installato
2. Controlla che il processo sia supportato
3. Esegui il test standalone per debugging
