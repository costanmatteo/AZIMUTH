# 🧠 Spiegazione SCM (Structural Causal Model)

## Cosa fa esattamente il file scm.py?

Il file `scm.py` è un **motore per modelli causali strutturali**. In pratica:

### 1. Definisci relazioni causa-effetto

Esempio semplice:
```
Temperatura → Vendite_Gelato
```

In SCM scrivi:
```python
NodeSpec("Temperatura", [], "eps_Temperatura")  # Variabile indipendente
NodeSpec("Vendite", ["Temperatura"], "5 * Temperatura + eps_Vendite")
```

Significa: `Vendite = 5 × Temperatura + rumore_casuale`

### 2. Genera dati che rispettano queste relazioni

Quando chiami `sample(n=1000)`, genera 1000 righe dove la relazione è sempre rispettata!

## 📐 Anatomia di un dataset SCM

Vediamo il dataset corrente `one_to_one_ct`:

```python
# File: scm_ds/datasets.py

ds_scm_1_to_1_ct = SCMDataset(
    name="one-to-one_with_crosstalk",

    # === DEFINIZIONE STRUTTURA CAUSALE ===
    specs=[
        # Parents (variabili esogene - non dipendono da nulla)
        NodeSpec("P1", [], "eps_P1"),  # P1 = rumore_P1
        NodeSpec("P2", [], "eps_P2"),
        NodeSpec("P3", [], "eps_P3"),
        NodeSpec("P4", [], "eps_P4"),
        NodeSpec("P5", [], "eps_P5"),

        # Children (dipendono dai parents)
        NodeSpec("C1", ["P1"], "P1 + eps_C1"),  # C1 = P1 + rumore_C1
        NodeSpec("C2", ["P2"], "P2 + eps_C2"),
        NodeSpec("C3", ["P3"], "P3 + eps_C3"),
        NodeSpec("C4", ["P4"], "P4 + eps_C4"),
        NodeSpec("C5", ["P5"], "P5 + eps_C5"),

        # Output finale (dipende da tutti i children)
        NodeSpec("Y", ["C1","C2","C3","C4","C5"],
                 "C1 + C2 + C3 + C4 + C5 + eps_Y"),
    ],

    # === PARAMETRI (opzionali) ===
    params={},  # Pesi/coefficienti parametrici

    # === DISTRIBUZIONE DEL RUMORE ===
    singles={
        "P1": lambda rng,n: rng.standard_normal(n),  # Rumore normale
        "P2": lambda rng,n: rng.standard_normal(n),
        # ... etc per tutti i nodi
    },

    # === INPUT/OUTPUT PER ML ===
    input_labels=["P1","P2","P3","P4","P5","C1","C2","C3","C4","C5"],
    target_labels=["Y"]
)
```

## 🔧 Come modificare la generazione dei dati

### Esempio 1: Cambiare i coefficienti

**Scenario**: Voglio che C1 abbia peso 2× invece di 1×

```python
# PRIMA:
NodeSpec("C1", ["P1"], "P1 + eps_C1")

# DOPO:
NodeSpec("C1", ["P1"], "2 * P1 + eps_C1")  # ← C1 ora è il doppio di P1
```

### Esempio 2: Aggiungere interazioni

**Scenario**: Voglio che Y dipenda anche dall'interazione tra C1 e C2

```python
# PRIMA:
NodeSpec("Y", ["C1","C2","C3","C4","C5"],
         "C1 + C2 + C3 + C4 + C5 + eps_Y")

# DOPO:
NodeSpec("Y", ["C1","C2","C3","C4","C5"],
         "C1 + C2 + C3 + C4 + C5 + 0.5*C1*C2 + eps_Y")
#        ^^^^^^^^^^^^^^^^^^^^^^ interazione!
```

### Esempio 3: Cambiare distribuzione del rumore

**Scenario**: Voglio rumore uniforme invece di normale per P1

```python
# PRIMA:
singles={
    "P1": lambda rng,n: rng.standard_normal(n),  # Normale(0,1)
}

# DOPO:
singles={
    "P1": lambda rng,n: rng.uniform(-1, 1, n),  # Uniforme[-1,1]
}
```

### Esempio 4: Creare un dataset completamente nuovo

```python
# File: scm_ds/datasets.py
# Aggiungi questo alla fine

ds_temperature_icecream = SCMDataset(
    name="temperature_icecream",
    description="Temperatura causa vendite gelato",
    tags=["simple", "causal"],

    specs=[
        # Temperatura (esogena)
        NodeSpec("Temp", [], "eps_Temp"),

        # Vendite (dipende dalla temperatura)
        NodeSpec("Sales", ["Temp"], "5*Temp + 10 + eps_Sales"),
        #                             ^^^^^^^^^^^
        #                             Sales aumentano con Temp
    ],

    params={},

    singles={
        "Temp": lambda rng,n: rng.normal(25, 5, n),    # Temp ~ N(25°C, σ=5)
        "Sales": lambda rng,n: rng.normal(0, 2, n),     # Rumore vendite
    },

    groups=None,
    input_labels=["Temp"],
    target_labels=["Sales"]
)
```

Poi usa con:
```python
# configs/example_config.py
'scm': {
    'dataset_type': 'temperature_icecream'  # ← Il tuo nuovo dataset!
}
```

## 🎯 Esempi pratici completi

### Esempio A: Sistema con 3 variabili e relazione non-lineare

```python
ds_nonlinear = SCMDataset(
    name="nonlinear_example",
    description="Relazione quadratica",
    tags=None,

    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Z", [], "eps_Z"),
        NodeSpec("Y", ["X", "Z"], "X**2 + 2*Z + eps_Y"),
        #                          ^^^^^ relazione quadratica!
    ],

    params={},

    singles={
        "X": lambda rng,n: rng.uniform(-2, 2, n),  # X ∈ [-2, 2]
        "Z": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 0.5, n),  # Poco rumore
    },

    groups=None,
    input_labels=["X", "Z"],
    target_labels=["Y"]
)
```

### Esempio B: Sistema con parametri configurabili

```python
ds_parametric = SCMDataset(
    name="parametric_example",
    description="Coefficienti parametrici",
    tags=None,

    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Y", ["X"], "w*X + b + eps_Y"),
        #                     ^   ^
        #                     parametri configurabili
    ],

    params={
        "w": 3.5,   # ← Peso
        "b": 1.2,   # ← Bias
    },

    singles={
        "X": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 0.1, n),
    },

    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)
```

### Esempio C: Variabili con rumore correlato

```python
from scm import GroupNoise

ds_correlated = SCMDataset(
    name="correlated_noise",
    description="Due variabili con rumore correlato",
    tags=None,

    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Y", [], "eps_Y"),  # Y ha rumore correlato con X!
        NodeSpec("Z", ["X", "Y"], "X + Y + eps_Z"),
    ],

    params={},

    singles={
        "Z": lambda rng,n: rng.standard_normal(n),
    },

    groups=[
        GroupNoise(
            nodes=("X", "Y"),  # ← X e Y hanno rumore correlato
            sampler=lambda rng,n: rng.multivariate_normal(
                mean=[0, 0],
                cov=[[1.0, 0.7],   # ← Correlazione 0.7
                     [0.7, 1.0]],
                size=n
            )
        )
    ],

    input_labels=["X", "Y"],
    target_labels=["Z"]
)
```

## 🔬 Testare le tue modifiche

Dopo aver modificato un dataset:

```python
# test_my_scm.py
import sys
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_temperature_icecream  # ← Il tuo dataset

# Genera campioni
df = ds_temperature_icecream.sample(n=100, seed=42)

# Verifica relazioni
print(df.head())
print(df.describe())

# Verifica correlazione
import matplotlib.pyplot as plt
plt.scatter(df['Temp'], df['Sales'])
plt.xlabel('Temperatura')
plt.ylabel('Vendite')
plt.show()
```

## 📊 Componenti chiave di NodeSpec

```python
NodeSpec(
    name="Y",              # Nome della variabile
    parents=["X", "Z"],    # Variabili da cui dipende (lista)
    expr="2*X + Z + eps_Y" # Equazione strutturale (stringa)
)
```

### Sintassi dell'equazione:
- **Variabili parent**: usa i nomi direttamente (`X`, `Z`)
- **Rumore**: `eps_<nome>` (es: `eps_Y` per Y)
- **Operatori**: `+`, `-`, `*`, `/`, `**` (potenza)
- **Funzioni**: `sqrt(X)`, `exp(X)`, `log(X)`, etc (SymPy)
- **Parametri**: definiti in `params` (es: `w`, `b`)

## 🎓 Workflow completo per creare un nuovo SCM

### Step 1: Progetta la struttura causale

Disegna il grafo:
```
Input1 → Middle → Output
Input2 ↗
```

### Step 2: Traduci in NodeSpec

```python
specs=[
    NodeSpec("Input1", [], "eps_Input1"),
    NodeSpec("Input2", [], "eps_Input2"),
    NodeSpec("Middle", ["Input1", "Input2"], "Input1 + 0.5*Input2 + eps_Middle"),
    NodeSpec("Output", ["Middle"], "2*Middle + eps_Output"),
]
```

### Step 3: Definisci il rumore

```python
singles={
    "Input1": lambda rng,n: rng.standard_normal(n),
    "Input2": lambda rng,n: rng.standard_normal(n),
    "Middle": lambda rng,n: rng.normal(0, 0.5, n),
    "Output": lambda rng,n: rng.normal(0, 0.1, n),
}
```

### Step 4: Specifica input/output

```python
input_labels=["Input1", "Input2"],
target_labels=["Output"]
```

### Step 5: Aggiungi a datasets.py

```python
# In scm_ds/datasets.py
ds_my_new_dataset = SCMDataset(
    name="my_new_dataset",
    description="...",
    tags=None,
    specs=[...],
    params={},
    singles={...},
    groups=None,
    input_labels=[...],
    target_labels=[...]
)
```

### Step 6: Registra nel loader

```python
# In src/data/preprocessing.py, funzione generate_scm_data()

if dataset_type == 'one_to_one_ct':
    scm_dataset = ds_scm_1_to_1_ct
elif dataset_type == 'my_new_dataset':  # ← Aggiungi questa linea
    from datasets import ds_my_new_dataset
    scm_dataset = ds_my_new_dataset
else:
    raise ValueError(f"Unknown SCM dataset type: {dataset_type}")
```

### Step 7: Usa nel config

```python
'scm': {
    'dataset_type': 'my_new_dataset'  # ← Usa il tuo dataset!
}
```

## 🚀 Quick Reference

### Equazioni comuni

```python
# Lineare
"a*X + b*Y + eps_Z"

# Quadratica
"X**2 + Y + eps_Z"

# Interazione
"X + Y + X*Y + eps_Z"

# Logistica (non-lineare)
"1 / (1 + exp(-X)) + eps_Z"

# Polinomiale
"a*X**3 + b*X**2 + c*X + eps_Z"

# Soglia/ReLU
"max(0, X) + eps_Z"  # Richiede: from sympy import Max
```

### Distribuzioni rumore comuni

```python
# Normale
lambda rng,n: rng.standard_normal(n)
lambda rng,n: rng.normal(mean, std, n)

# Uniforme
lambda rng,n: rng.uniform(low, high, n)

# Esponenziale
lambda rng,n: rng.exponential(scale, n)

# Poisson
lambda rng,n: rng.poisson(lam, n)

# Beta
lambda rng,n: rng.beta(a, b, n)
```

## 💡 Tips

1. **Start simple**: Inizia con 2-3 variabili
2. **Verifica correlazioni**: Controlla che i dati generati abbiano senso
3. **Plot dei dati**: Visualizza sempre i dati generati
4. **Seed fisso**: Usa seed fisso per debugging (es: `seed=42`)
5. **Scala appropriata**: Assicurati che le scale delle variabili siano ragionevoli

## 🐛 Debugging

Se qualcosa non funziona:

```python
# 1. Importa direttamente
from scm_ds.datasets import ds_my_dataset

# 2. Genera pochi campioni
df = ds_my_dataset.sample(n=10, seed=42)

# 3. Stampa tutto
print(df)
print(df.describe())
print(df.corr())

# 4. Verifica manualmente le equazioni
# Se Y = X + Z, allora Y dovrebbe essere circa X + Z
print("X + Z =", df['X'] + df['Z'])
print("Y =", df['Y'])
```
