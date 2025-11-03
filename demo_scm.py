#!/usr/bin/env python3
"""
🎯 Demo SCM - Genera e visualizza dataset sintetico
Esegui: python demo_scm.py
"""

import sys
from pathlib import Path
import numpy as np

# Setup paths
sys.path.insert(0, str(Path('uncertainty_predictor/scm_ds')))

print("=" * 70)
print("🎯 DEMO: Generazione Dataset SCM")
print("=" * 70)

# Import SCM dataset
from datasets import ds_scm_1_to_1_ct

print("\n📋 Info Dataset SCM 'one_to_one_ct':")
print(f"   Input features:  {ds_scm_1_to_1_ct.input_labels}")
print(f"   Output features: {ds_scm_1_to_1_ct.target_labels}")
print(f"   Totale features: {len(ds_scm_1_to_1_ct.input_labels) + len(ds_scm_1_to_1_ct.target_labels)}")

# Genera campioni
n_samples = 1000
seed = 42

print(f"\n⚙️  Generando {n_samples} campioni (seed={seed})...")
df = ds_scm_1_to_1_ct.sample(n=n_samples, seed=seed)

print(f"\n✅ Dataset generato con successo!")
print(f"   Shape: {df.shape}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Mostra statistiche
print("\n📊 Statistiche del dataset:")
print("-" * 70)
print(df.describe().round(3))

# Mostra prime righe
print("\n📋 Prime 5 righe:")
print("-" * 70)
print(df.head())

# Analisi correlazioni
print("\n🔗 Correlazioni con output Y:")
print("-" * 70)
correlations = df.corr()['Y'].sort_values(ascending=False)
for var, corr in correlations.items():
    if var != 'Y':
        bar = '█' * int(abs(corr) * 20)
        print(f"   {var:5s}: {corr:+.3f}  {bar}")

# Verifica relazioni causali
print("\n🔬 Verifica relazioni causali:")
print("-" * 70)
print("   Parent → Child (correlazione attesa: alta)")
pairs = [('P1', 'C1'), ('P2', 'C2'), ('P3', 'C3'), ('P4', 'C4'), ('P5', 'C5')]
for parent, child in pairs:
    corr = df[[parent, child]].corr().iloc[0, 1]
    status = '✓' if corr > 0.8 else '✗'
    print(f"   {status} {parent} → {child}: {corr:.3f}")

print("\n   Children → Y (correlazione attesa: alta)")
for child in ['C1', 'C2', 'C3', 'C4', 'C5']:
    corr = df[[child, 'Y']].corr().iloc[0, 1]
    status = '✓' if corr > 0.3 else '✗'
    print(f"   {status} {child} → Y: {corr:.3f}")

# Preparazione per ML
print("\n🤖 Preparazione per Machine Learning:")
print("-" * 70)

X = df[ds_scm_1_to_1_ct.input_labels].values
y = df[ds_scm_1_to_1_ct.target_labels].values

print(f"   X shape: {X.shape}  (features: {X.shape[1]})")
print(f"   y shape: {y.shape}  (targets: {y.shape[1]})")
print(f"   X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"   y range: [{y.min():.2f}, {y.max():.2f}]")

# Salva dataset (opzionale)
save_path = Path('uncertainty_predictor/demo_scm_data.csv')
df.to_csv(save_path, index=False)
print(f"\n💾 Dataset salvato in: {save_path}")

print("\n" + "=" * 70)
print("✅ Demo completata!")
print("=" * 70)
print("\n📖 Per usare SCM nel training:")
print("   1. Apri: uncertainty_predictor/configs/example_config.py")
print("   2. Imposta: 'csv_path': None")
print("   3. Imposta: 'use_scm': True")
print("   4. Esegui: python uncertainty_predictor/train.py")
print("\n📚 Guida completa: uncertainty_predictor/SCM_USAGE_GUIDE.md")
print("=" * 70)
