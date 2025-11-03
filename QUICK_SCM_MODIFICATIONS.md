# 🚀 Modifiche SCM Veloci - Copy & Paste

## 🎯 Modifiche al dataset esistente (one_to_one_ct)

### 1. Raddoppiare l'effetto dei Parents sui Children

**File**: `uncertainty_predictor/scm_ds/datasets.py`

```python
# PRIMA:
NodeSpec("C1", ["P1"], "P1 + eps_C1"),

# DOPO:
NodeSpec("C1", ["P1"], "2 * P1 + eps_C1"),  # ← C1 ora è il doppio
```

Applica a tutti: C1, C2, C3, C4, C5

### 2. Aggiungere interazioni tra Children nell'output

```python
# PRIMA:
NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],
         "C1 + C2 + C3 + C4 + C5 + eps_Y"),

# DOPO:
NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],
         "C1 + C2 + C3 + C4 + C5 + 0.3*C1*C2 + 0.2*C3*C4 + eps_Y"),
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ interazioni!
```

### 3. Cambiare il rumore da normale a uniforme

```python
# PRIMA:
singles = {
    "P1": lambda rng,n: rng.standard_normal(n),
}

# DOPO:
singles = {
    "P1": lambda rng,n: rng.uniform(-1, 1, n),  # ← Rumore uniforme
}
```

### 4. Aggiungere un bias ai Children

```python
# PRIMA:
NodeSpec("C1", ["P1"], "P1 + eps_C1"),

# DOPO:
NodeSpec("C1", ["P1"], "P1 + 10 + eps_C1"),  # ← Offset di +10
```

### 5. Ridurre il rumore (dati più "puliti")

```python
# PRIMA:
singles = {
    "Y": lambda rng,n: rng.standard_normal(n),  # σ = 1.0
}

# DOPO:
singles = {
    "Y": lambda rng,n: rng.normal(0, 0.1, n),  # σ = 0.1 (molto meno rumore!)
}
```

## 🆕 Dataset nuovi - Copy & Paste

### Dataset A: Sistema semplice 2→1

**Aggiungi a `datasets.py`:**

```python
ds_simple_2to1 = SCMDataset(
    name="simple_2to1",
    description="Due input, un output - relazione lineare",
    tags=None,
    specs=[
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("Y", ["X1", "X2"], "3*X1 + 2*X2 + eps_Y"),
    ],
    params={},
    singles={
        "X1": lambda rng,n: rng.standard_normal(n),
        "X2": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 0.5, n),
    },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels=["Y"]
)
```

**Modifica `preprocessing.py` (funzione `generate_scm_data`):**

```python
# Aggiungi dopo la linea "if dataset_type == 'one_to_one_ct':"
elif dataset_type == 'simple_2to1':
    from datasets import ds_simple_2to1
    scm_dataset = ds_simple_2to1
```

**Usa nel config:**

```python
'scm': {
    'dataset_type': 'simple_2to1'  # ← Il tuo nuovo dataset!
}
```

### Dataset B: Relazione quadratica

```python
ds_quadratic = SCMDataset(
    name="quadratic",
    description="Relazione quadratica",
    tags=None,
    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Y", ["X"], "X**2 + 2*X + 5 + eps_Y"),  # Y = X² + 2X + 5
    ],
    params={},
    singles={
        "X": lambda rng,n: rng.uniform(-3, 3, n),  # X ∈ [-3, 3]
        "Y": lambda rng,n: rng.normal(0, 0.5, n),
    },
    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)
```

### Dataset C: 5 input → 1 output (senza layer intermedio)

```python
ds_5to1_direct = SCMDataset(
    name="5to1_direct",
    description="Cinque input diretti all'output",
    tags=None,
    specs=[
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("X3", [], "eps_X3"),
        NodeSpec("X4", [], "eps_X4"),
        NodeSpec("X5", [], "eps_X5"),
        NodeSpec("Y", ["X1", "X2", "X3", "X4", "X5"],
                 "2*X1 + 1.5*X2 + X3 + 0.5*X4 + 0.3*X5 + eps_Y"),
    ],
    params={},
    singles={
        "X1": lambda rng,n: rng.standard_normal(n),
        "X2": lambda rng,n: rng.standard_normal(n),
        "X3": lambda rng,n: rng.standard_normal(n),
        "X4": lambda rng,n: rng.standard_normal(n),
        "X5": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 1, n),
    },
    groups=None,
    input_labels=["X1", "X2", "X3", "X4", "X5"],
    target_labels=["Y"]
)
```

### Dataset D: Con parametri configurabili

```python
ds_parametric = SCMDataset(
    name="parametric",
    description="Coefficienti facili da cambiare",
    tags=None,
    specs=[
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("Y", ["X1", "X2"], "w1*X1 + w2*X2 + bias + eps_Y"),
        #                            ^^   ^^   ^^^^
        #                            Parametri configurabili
    ],
    params={
        "w1": 3.5,     # ← Cambia questo per modificare peso di X1
        "w2": 1.2,     # ← Cambia questo per modificare peso di X2
        "bias": 10.0,  # ← Cambia questo per offset
    },
    singles={
        "X1": lambda rng,n: rng.standard_normal(n),
        "X2": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 0.1, n),
    },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels=["Y"]
)
```

## 🧪 Testing veloce

Dopo ogni modifica, testa con:

```python
# test_quick.py
import sys
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_scm_1_to_1_ct  # ← Cambia con il tuo dataset

df = ds_scm_1_to_1_ct.sample(n=100, seed=42)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nPrime righe:")
print(df.head())
print("\nStatistiche:")
print(df.describe())
```

## 🎛️ Configurazioni comuni

### Config 1: Test veloce
```python
'scm': {
    'n_samples': 1000,   # Pochi campioni
    'seed': 42,
    'dataset_type': 'one_to_one_ct'
}
```

### Config 2: Training serio
```python
'scm': {
    'n_samples': 50000,  # Molti campioni
    'seed': 42,
    'dataset_type': 'one_to_one_ct'
}
```

### Config 3: Debugging (sempre stesso dataset)
```python
'scm': {
    'n_samples': 500,
    'seed': 123,  # ← Seed fisso
    'dataset_type': 'one_to_one_ct'
},
'random_state': 123,  # ← Stesso seed per split
'misc': {
    'random_seed': 123  # ← Stesso seed per training
}
```

## 📝 Checklist dopo una modifica

- [ ] Modifica fatta in `datasets.py`
- [ ] Testato con `test_quick.py`
- [ ] Verificato che correlazioni hanno senso
- [ ] (Se nuovo dataset) Aggiunto a `preprocessing.py`
- [ ] (Se nuovo dataset) Aggiornato config con nuovo nome
- [ ] Eseguito `python train.py` per test finale

## 💡 Tips

1. **Start small**: Prima testa con `n_samples=100`
2. **Verifica correlazioni**: Usa `df.corr()` per controllare
3. **Plot sempre**: Visualizza i dati generati
4. **Seed fisso**: Usa `seed=42` per debugging
5. **Incrementale**: Fai una modifica alla volta

## 🐛 Troubleshooting

### "NameError: name 'max' is not defined"
```python
# NON funziona:
"max(0, X) + eps_Y"

# Funziona:
from sympy import Max
"Max(0, X) + eps_Y"
```

### "ValueError: Unknown SCM dataset type"
Hai dimenticato di aggiungere il dataset a `preprocessing.py`:
```python
elif dataset_type == 'il_tuo_nome':
    from datasets import ds_il_tuo_nome
    scm_dataset = ds_il_tuo_nome
```

### Correlazioni strane nei dati generati
Controlla:
1. Equazioni strutturali corrette?
2. Rumore troppo grande? (prova `sigma=0.1` invece di `1.0`)
3. Scale delle variabili appropriate?
