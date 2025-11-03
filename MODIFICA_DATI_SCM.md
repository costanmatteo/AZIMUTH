# 🎯 Guida: Come modificare la generazione dati SCM

## 🎨 3 modi per personalizzare i dati

---

## **METODO 1: Modificare il dataset esistente** ⭐ PIÙ SEMPLICE

### Cosa modificare: `uncertainty_predictor/scm_ds/datasets.py`

### 🔧 Modifiche rapide:

#### A) Cambiare i coefficienti

```python
# PRIMA (linea 22):
NodeSpec("C1", ["P1"], "P1 + eps_C1"),

# DOPO - P1 ha 3 volte più peso:
NodeSpec("C1", ["P1"], "3 * P1 + eps_C1"),
```

#### B) Aggiungere interazioni

```python
# PRIMA (linea 28):
NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],
         "C1 + C2 + C3 + C4 + C5 + eps_Y"),

# DOPO - Aggiungi interazioni tra C1 e C2:
NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],
         "C1 + C2 + C3 + C4 + C5 + 0.5*C1*C2 + eps_Y"),
```

#### C) Ridurre il rumore

```python
# PRIMA (linea 48):
"Y": lambda rng,n: rng.standard_normal(n),  # Rumore std=1.0

# DOPO - Rumore 10 volte minore:
"Y": lambda rng,n: rng.normal(0, 0.1, n),  # Rumore std=0.1
```

#### D) Cambiare distribuzione del rumore

```python
# PRIMA (linea 38):
"P1": lambda rng,n: rng.standard_normal(n),  # Normale(0,1)

# DOPO - Uniforme tra -2 e 2:
"P1": lambda rng,n: rng.uniform(-2, 2, n),

# Oppure Normale con media e std personalizzati:
"P1": lambda rng,n: rng.normal(10, 2, n),  # Media=10, Std=2
```

### ✅ Test dopo la modifica:

```bash
cd uncertainty_predictor
python train.py
```

---

## **METODO 2: Creare un nuovo dataset personalizzato** ⭐ PIÙ FLESSIBILE

### Step 1: Aggiungi il dataset a `datasets.py`

Apri `uncertainty_predictor/scm_ds/datasets.py` e **aggiungi PRIMA della riga 58**:

```python
# Il tuo nuovo dataset
ds_mio_dataset = SCMDataset(
    name="mio_dataset",
    description="Descrizione del mio dataset",
    tags=None,

    specs=[
        # Definisci le variabili e le relazioni
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("Y", ["X1", "X2"], "2*X1 + X2 + eps_Y"),
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

### Step 2: Registra in `preprocessing.py`

Apri `uncertainty_predictor/src/data/preprocessing.py`, trova la linea ~225 e **decomma** queste righe:

```python
# PRIMA (commentato):
# elif dataset_type == 'simple_3to1':
#     from datasets import ds_simple_3to1
#     scm_dataset = ds_simple_3to1

# DOPO (decommentato e modificato):
elif dataset_type == 'mio_dataset':
    from datasets import ds_mio_dataset
    scm_dataset = ds_mio_dataset
```

### Step 3: Usa nel config

Apri `configs/example_config.py`:

```python
'scm': {
    'n_samples': 5000,
    'seed': 42,
    'dataset_type': 'mio_dataset'  # ← Il tuo nuovo dataset!
}
```

### Step 4: Test

```bash
cd uncertainty_predictor
python train.py
```

---

## **METODO 3: Usa un template pronto** ⭐ PIÙ VELOCE

Ho preparato 5 template in `example_custom_datasets.py`:

### Template 1: Dataset semplice (3→1)

**Caratteristiche:**
- 3 input indipendenti (Temperatura, Umidità, Pressione)
- 1 output (Consumo)
- Relazione lineare con pesi diversi

**Come usare:**
1. Copia il codice di `ds_simple_3to1` da `example_custom_datasets.py`
2. Incollalo in `datasets.py` (prima riga 58)
3. Aggiungi in `preprocessing.py`:
   ```python
   elif dataset_type == 'simple_3to1':
       from datasets import ds_simple_3to1
       scm_dataset = ds_simple_3to1
   ```
4. Config: `'dataset_type': 'simple_3to1'`

### Template 2: Dataset non-lineare

**Caratteristiche:**
- Relazione quadratica: Y = X² + 2X + 5

**Come usare:**
1. Copia `ds_nonlinear`
2. Aggiungi a `datasets.py`
3. Registra in `preprocessing.py`
4. Config: `'dataset_type': 'nonlinear'`

### Template 3: Dataset gerarchico

**Caratteristiche:**
- Multi-layer: Input → Features intermedie → Output
- Simula feature engineering automatico

### Template 4: Dataset parametrico

**Caratteristiche:**
- Pesi facilmente modificabili tramite `params`
- Basta cambiare `w1`, `w2`, `bias` per modificare il comportamento

### Template 5: Dataset semplificato

**Caratteristiche:**
- Come l'attuale ma con 3 parent invece di 5
- Più veloce da trainare

---

## 📊 Esempi di modifiche comuni

### Caso d'uso 1: Voglio dati più "puliti" (meno rumorosi)

```python
singles={
    # Tutti i rumoris con std=0.1 invece di 1.0
    "P1": lambda rng,n: rng.normal(0, 0.1, n),
    "P2": lambda rng,n: rng.normal(0, 0.1, n),
    # ...etc
    "Y": lambda rng,n: rng.normal(0, 0.05, n),  # Output ancora meno rumore
}
```

### Caso d'uso 2: Voglio relazioni non-lineari

```python
# Quadratica
NodeSpec("Y", ["X"], "X**2 + 2*X + eps_Y"),

# Esponenziale
NodeSpec("Y", ["X"], "exp(0.5*X) + eps_Y"),

# Con soglia (ReLU-like)
NodeSpec("Y", ["X"], "Piecewise((0, X < 0), (X, True)) + eps_Y"),
```

### Caso d'uso 3: Voglio più interazioni tra variabili

```python
NodeSpec("Y", ["X1", "X2", "X3"],
         "X1 + X2 + X3 + 0.5*X1*X2 + 0.3*X2*X3 + 0.2*X1*X3 + eps_Y"),
```

### Caso d'uso 4: Voglio dati con scala specifica

```python
singles={
    # Temperatura in range realistico
    "Temp": lambda rng,n: rng.normal(20, 5, n),  # ~20°C ± 5

    # Percentuali
    "Humidity": lambda rng,n: np.clip(rng.normal(60, 15, n), 0, 100),

    # Valori positivi (esponenziale)
    "Price": lambda rng,n: rng.exponential(50, n),  # Media ~50
}
```

---

## 🧪 Testing delle modifiche

### Test 1: Genera pochi campioni

```python
# test_scm_quick.py
import sys
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_scm_1_to_1_ct  # ← Cambia con il tuo dataset

df = ds_scm_1_to_1_ct.sample(n=100, seed=42)

print("Shape:", df.shape)
print("\nPrime righe:")
print(df.head())
print("\nStatistiche:")
print(df.describe())
print("\nCorrelazioni:")
print(df.corr().round(3))
```

### Test 2: Verifica relazioni causali

```python
import matplotlib.pyplot as plt

# Plot correlazione tra parent e child
plt.scatter(df['P1'], df['C1'])
plt.xlabel('P1')
plt.ylabel('C1')
plt.title('P1 → C1 (dovrebbe essere alta)')
plt.show()

# Verifica numerica
corr = df[['P1', 'C1']].corr().iloc[0,1]
print(f"Correlazione P1→C1: {corr:.3f}")  # Dovrebbe essere alta (>0.7)
```

### Test 3: Training completo

```bash
cd uncertainty_predictor
python train.py
```

---

## 🎨 Anatomia di un NodeSpec

```python
NodeSpec(
    name="Y",              # Nome variabile
    parents=["X1", "X2"],  # Variabili da cui dipende
    expr="2*X1 + X2 + eps_Y"  # Equazione strutturale
)
```

### Sintassi equazioni:

```python
# Lineare
"a*X + b*Y + eps_Z"

# Quadratica
"X**2 + Y + eps_Z"

# Interazione
"X*Y + eps_Z"

# Esponenziale
"exp(X) + eps_Z"

# Logaritmo
"log(abs(X) + 1) + eps_Z"  # +1 per evitare log(0)

# Funzioni trigonometriche
"sin(X) + cos(Y) + eps_Z"

# Combinazioni
"2*X**2 + 3*Y + 0.5*X*Y + eps_Z"
```

### Distribuzioni rumore comuni:

```python
# Normale standard
lambda rng,n: rng.standard_normal(n)

# Normale personalizzata
lambda rng,n: rng.normal(mean, std, n)

# Uniforme
lambda rng,n: rng.uniform(low, high, n)

# Esponenziale (solo valori positivi)
lambda rng,n: rng.exponential(scale, n)

# Poisson (conteggi)
lambda rng,n: rng.poisson(lam, n)

# Beta (valori in [0,1])
lambda rng,n: rng.beta(a, b, n)

# Binomiale
lambda rng,n: rng.binomial(n_trials, p, n)
```

---

## ✅ Checklist modifiche

Prima di eseguire il training:

- [ ] Ho modificato `datasets.py` con il mio dataset?
- [ ] Ho aggiunto il dataset in `preprocessing.py`?
- [ ] Ho aggiornato `config.py` con il nome corretto?
- [ ] Ho testato con pochi campioni (n=100)?
- [ ] Le correlazioni hanno senso?
- [ ] Ho verificato che non ci siano errori di sintassi?

---

## 🐛 Errori comuni

### Errore: "name 'ds_mio_dataset' is not defined"

**Causa:** Non hai importato il dataset in `preprocessing.py`

**Fix:** Aggiungi:
```python
elif dataset_type == 'mio_dataset':
    from datasets import ds_mio_dataset
    scm_dataset = ds_mio_dataset
```

### Errore: "Unknown SCM dataset type"

**Causa:** Il nome in config non corrisponde

**Fix:** Verifica che:
- Config: `'dataset_type': 'mio_dataset'`
- preprocessing.py: `if dataset_type == 'mio_dataset':`

Devono essere identici!

### Errore: "NameError: name 'exp' is not defined"

**Causa:** Funzioni speciali vanno importate da sympy

**Fix:** All'inizio di `datasets.py` aggiungi:
```python
from sympy import exp, log, sin, cos, sqrt, Max, Min
```

Poi usa:
```python
NodeSpec("Y", ["X"], "exp(X) + eps_Y")
```

### Warning: correlazioni molto basse

**Causa:** Troppo rumore

**Fix:** Riduci il rumore:
```python
"Y": lambda rng,n: rng.normal(0, 0.1, n)  # Invece di 1.0
```

---

## 💡 Tips

1. **Start simple**: Inizia con 2-3 variabili
2. **Test con n=100**: Prima di generare migliaia di campioni
3. **Verifica correlazioni**: Usa `df.corr()` per controllo
4. **Plot sempre**: Visualizza le relazioni
5. **Incrementale**: Una modifica alla volta
6. **Backup**: Salva una copia del dataset originale

---

## 📚 Risorse

- **Esempi pronti**: `example_custom_datasets.py`
- **Guida completa**: `SCM_EXPLAINED.md`
- **Quick reference**: `QUICK_SCM_MODIFICATIONS.md`
- **Setup Windows**: `WINDOWS_SETUP.md`

---

## 🎯 Quick Start per i più impazienti

**Modificare il dataset esistente (30 secondi):**

1. Apri `uncertainty_predictor/scm_ds/datasets.py`
2. Linea 22: Cambia `"P1 + eps_C1"` → `"3 * P1 + eps_C1"`
3. Salva
4. `python train.py`

Done! ✅
