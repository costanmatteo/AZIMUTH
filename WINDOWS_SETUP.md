# 🪟 Guida Windows - Setup SCM

## Problema comune su Windows

```
ModuleNotFoundError: No module named 'graphviz'
```

## ✅ Soluzione (già applicata!)

Il codice è stato modificato per rendere graphviz **opzionale**. Ora funziona anche senza!

## 🚀 Quick Fix - Installa solo le dipendenze essenziali

```bash
pip install sympy pandas numpy scikit-learn
```

Poi esegui:
```bash
cd uncertainty_predictor
python train.py
```

✅ Funzionerà! (senza la visualizzazione del grafo)

## 🎨 (Opzionale) Installare graphviz per i grafi

Se vuoi anche la visualizzazione dei grafi causali:

### Metodo 1: Con Chocolatey (consigliato)

1. Installa Chocolatey: https://chocolatey.org/install
2. In PowerShell (come Admin):
   ```bash
   choco install graphviz
   ```
3. Installa il pacchetto Python:
   ```bash
   pip install graphviz
   ```

### Metodo 2: Manuale

1. Scarica Graphviz: https://graphviz.org/download/
2. Installa in `C:\Program Files\Graphviz`
3. Aggiungi al PATH:
   - Apri "Variabili d'ambiente"
   - Aggiungi `C:\Program Files\Graphviz\bin` al PATH
4. Riavvia il terminale
5. Installa il pacchetto Python:
   ```bash
   pip install graphviz
   ```

### Metodo 3: Con Anaconda/Conda

```bash
conda install python-graphviz
```

## 📋 Verifica installazione

```bash
# Controlla sympy (essenziale)
python -c "import sympy; print('✓ sympy OK')"

# Controlla graphviz (opzionale)
python -c "import graphviz; print('✓ graphviz OK')"
```

## ⚡ Installa tutto da requirements.txt

```bash
cd uncertainty_predictor
pip install -r requirements.txt
```

**Note:** Anche se `requirements.txt` include graphviz, il codice funzionerà comunque senza!

## 🧪 Test rapido

```python
# test_scm_windows.py
import sys
sys.path.insert(0, 'scm_ds')

print("Testing SCM import...")
from datasets import ds_scm_1_to_1_ct
print("✓ SCM imported successfully!")

print("\nGenerating 10 samples...")
df = ds_scm_1_to_1_ct.sample(n=10, seed=42)
print(f"✓ Generated {len(df)} samples")
print(f"✓ Columns: {list(df.columns)}")
print("\n✅ SCM works on Windows!")
```

Esegui:
```bash
cd uncertainty_predictor
python test_scm_windows.py
```

## 🐛 Troubleshooting Windows

### PowerShell ExecutionPolicy Error
Se vedi "cannot be loaded because running scripts is disabled":
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip non riconosciuto
Usa:
```bash
python -m pip install sympy
```

### Path troppo lungo (errore Windows)
Abilita long paths:
1. Win+R → `gpedit.msc`
2. Computer Configuration → Administrative Templates → System → Filesystem
3. Enable "Enable Win32 long paths"

### Errore SSL/Certificate
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org sympy
```

## 📊 Differenze con/senza graphviz

| Feature | Con graphviz | Senza graphviz |
|---------|--------------|----------------|
| Generazione dati SCM | ✅ | ✅ |
| Training modello | ✅ | ✅ |
| Preprocessing | ✅ | ✅ |
| Tutte le funzionalità ML | ✅ | ✅ |
| Grafo causale (PDF) | ✅ | ⚠️ Saltato |

**Conclusione**: Graphviz è utile per debugging, ma **non necessario** per il training!

## 🎯 Workflow consigliato su Windows

1. **Installazione minima** (per iniziare subito):
   ```bash
   pip install sympy pandas numpy scikit-learn torch
   ```

2. **Test SCM**:
   ```bash
   python demo_scm.py
   ```

3. **Training**:
   ```bash
   cd uncertainty_predictor
   python train.py
   ```

4. **(Opzionale) Aggiungi graphviz più tardi** se vuoi i grafi

## ✅ Checklist

- [x] sympy installato?
- [x] pandas, numpy installati?
- [ ] graphviz installato? (opzionale)
- [x] train.py funziona?

Se hai checkato i primi 3, sei a posto! 🎉
