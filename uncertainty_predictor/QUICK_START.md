# 🚀 Quick Start - In 3 Passi

## Opzione A: Test Veloce con Dati Auto-Generati

```bash
# 1. Installa dipendenze (solo prima volta)
cd uncertainty_predictor
pip install -r requirements.txt

# 2. Training
python train.py

# 3. Guarda i risultati
ls checkpoints_uncertainty/
# Apri: training_report.pdf
```

**Fatto in 30 secondi!** ✅

---

## Opzione B: Con i TUOI Dati

### 1️⃣ Prepara il CSV

Crea un file CSV tipo questo:

**my_data.csv**
```csv
input1,input2,input3,output
1.2,3.4,5.6,10.2
2.3,4.5,6.7,11.5
...
```

### 2️⃣ Configura

Apri `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'my_data.csv',                    # ← Il tuo file
        'input_columns': ['input1', 'input2', 'input3'],  # ← Le tue colonne input
        'output_columns': ['output'],                 # ← La tua colonna output
        # ... resto uguale
    }
}
```

### 3️⃣ Training

```bash
cd uncertainty_predictor
python train.py
```

### 4️⃣ Risultati

Vai in `checkpoints_uncertainty/` e apri:
- 📊 `training_report.pdf` - Report completo
- 📈 `predictions_with_uncertainty.png` - Predizioni con bande
- 📉 `scatter_with_uncertainty.png` - Scatter plot

---

## 📝 Esempio Completo Passo-Passo

### Scenario: Dati di Produzione

```bash
# 1. Crea il file dati
cat > production.csv << EOF
temp,press,speed,quality
25.3,101.2,1500,95.2
26.1,102.5,1480,94.8
24.8,100.9,1520,96.1
EOF

# 2. Copia nella cartella
cp production.csv uncertainty_predictor/

# 3. Modifica config
cd uncertainty_predictor
nano configs/example_config.py
# Cambia:
#   'csv_path': 'production.csv'
#   'input_columns': ['temp', 'press', 'speed']
#   'output_columns': ['quality']

# 4. Training
python train.py

# 5. Vedi risultati
open checkpoints_uncertainty/training_report.pdf
```

---

## ❓ FAQ

**Q: Quanti dati servono?**
A: Minimo ~100 righe, ideale 1000+

**Q: Il CSV deve avere headers?**
A: Sì, la prima riga deve avere i nomi delle colonne

**Q: Posso usare più output?**
A: Sì! `'output_columns': ['output1', 'output2', 'output3']`

**Q: Quanto tempo ci vuole?**
A: 2-10 minuti a seconda dei dati (con CPU)

**Q: Come uso il modello dopo il training?**
A: Guarda `predict.py` per esempi

---

## 🎓 Cosa Ottieni

Il modello ti dà **2 valori** per ogni predizione:

1. **Valore predetto** (μ): La stima
2. **Incertezza** (σ²): Quanto è sicuro

**Esempio**:
```
Input: temp=25°C, press=101kPa, speed=1500rpm
Output:
  ✓ Qualità predetta: 95.2%
  ± Incertezza: ±2.1%
  → Intervallo 95%: [91.0%, 99.4%]
```

Questo ti permette di:
- ✅ Sapere quando fidarti delle predizioni
- ✅ Identificare casi anomali
- ✅ Prendere decisioni basate sul rischio

---

## 🔗 Documentazione Completa

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Guida dettagliata
- [README.md](README.md) - Documentazione tecnica
- [configs/example_config.py](configs/example_config.py) - Tutti i parametri

---

## 💡 Esempi Pronti

Abbiamo preparato un esempio con dati fittizi:

```bash
# Usa il config di esempio
python train.py --config configs/production_config.py
```

Questo usa il file `example_production_data.csv` già incluso!
