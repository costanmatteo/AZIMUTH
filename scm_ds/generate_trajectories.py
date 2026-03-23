"""
Generazione centralizzata di traiettorie di training per l'intera catena di processi.

Questo modulo fornisce:
- generate_training_trajectories(): genera un DataFrame con tutti gli input/output
  di tutti i processi e la colonna F (reliability) per ogni riga
- compute_F_numpy(): formula di F in numpy puro (single source of truth)
- get_casualit_view(): proiezione del DataFrame nella forma (X, y) per CausalIT
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_F_numpy(row: pd.Series, process_configs: list) -> float:
    """
    Calcola la reliability F per una singola riga del DataFrame.

    Formula:
        Per ogni processo i:
            Q_i = exp(-(Y_i - τ_i)² / s_i)
        dove τ_i è il target adattivo che dipende dagli output dei processi precedenti.

        F = Σ(w_i × Q_i) / Σ(w_i)

    Questa è la singola source of truth per la formula di F nel contesto dei dati
    di training. Reimplementa in numpy puro la stessa logica di
    reliability_function/src/compute_reliability.py e di
    controller_optimization/src/models/surrogate.py.

    Args:
        row: Una riga del DataFrame con tutte le colonne output dei processi.
        process_configs: Lista di dizionari di configurazione dei processi,
            ciascuno con chiavi:
            - 'name': nome del processo
            - 'output_labels': lista di nomi colonne output
            - Per processi fisici: si usano i config da PROCESS_TARGETS
            - Per processi ST: si usano 'surrogate_target', 'surrogate_scale',
              'surrogate_weight', e opzionalmente
              'surrogate_adaptive_coefficients'/'surrogate_adaptive_baselines'

    Returns:
        float: Valore di reliability F in [0, 1]
    """
    # Determina la configurazione di reliability per ogni processo.
    # Per processi ST usiamo i campi surrogate_* dal process_config.
    # Per processi fisici usiamo PROCESS_TARGETS.
    reliability_configs, process_order = _build_reliability_configs(process_configs)

    # Raccogli gli output di ogni processo dalla riga
    outputs = {}
    for pc in process_configs:
        name = pc['name']
        # Prendi il primo output label (i processi hanno tipicamente 1 output)
        output_col = f"{name}_{pc['output_labels'][0]}"
        if output_col in row.index:
            outputs[name] = row[output_col]

    # Calcola quality scores
    total_weighted_quality = 0.0
    total_weight = 0.0

    for process_name in process_order:
        if process_name not in outputs:
            continue

        cfg = reliability_configs.get(process_name, {})
        output_val = outputs[process_name]

        # Calcola target adattivo
        target = cfg.get('base_target', 0.0)
        for upstream_name, coeff in cfg.get('adaptive_coefficients', {}).items():
            if upstream_name in outputs:
                baseline = cfg.get('adaptive_baselines', {}).get(upstream_name, 0.0)
                target = target + coeff * (outputs[upstream_name] - baseline)

        scale = cfg.get('scale', 1.0)
        quality = np.exp(-((output_val - target) ** 2) / max(scale, 1e-8))

        weight = cfg.get('weight', 1.0)
        total_weighted_quality += quality * weight
        total_weight += weight

    if total_weight > 0:
        return total_weighted_quality / total_weight
    return 0.0


def _build_reliability_configs(process_configs: list) -> Tuple[dict, list]:
    """
    Costruisce la mappa di configurazione reliability a partire da process_configs.

    Per processi ST: usa surrogate_target/surrogate_scale/surrogate_weight.
    Per processi fisici: usa PROCESS_TARGETS da reliability_function.

    Returns:
        (reliability_configs, process_order): dizionario name->config e lista ordinata
    """
    process_order = [pc['name'] for pc in process_configs]

    # Controlla se sono processi ST (hanno surrogate_target)
    is_st = any('surrogate_target' in pc for pc in process_configs)

    if is_st:
        configs = {}
        for pc in process_configs:
            name = pc['name']
            entry = {
                'base_target': pc.get('surrogate_target', 0.0),
                'scale': pc.get('surrogate_scale', 1.0),
                'weight': pc.get('surrogate_weight', 1.0),
            }
            if 'surrogate_adaptive_coefficients' in pc:
                entry['adaptive_coefficients'] = pc['surrogate_adaptive_coefficients']
                entry['adaptive_baselines'] = pc.get('surrogate_adaptive_baselines', {})
            else:
                entry['adaptive_coefficients'] = {}
                entry['adaptive_baselines'] = {}
            configs[name] = entry
        return configs, process_order
    else:
        # Processi fisici: carica da reliability_function
        from reliability_function.configs.process_targets import PROCESS_CONFIGS
        configs = {}
        for pc in process_configs:
            name = pc['name']
            if name in PROCESS_CONFIGS:
                cfg = dict(PROCESS_CONFIGS[name])
                cfg.setdefault('adaptive_coefficients', {})
                cfg.setdefault('adaptive_baselines', {})
                configs[name] = cfg
        return configs, process_order


def generate_training_trajectories(
    process_configs: list,
    n_samples: int = 5000,
    seed: int = 42,
    noise_mode: str = 'active',
) -> pd.DataFrame:
    """
    Genera traiettorie di training complete per l'intera catena di processi.

    Per ogni processo, campiona n_samples righe dal suo SCM. Tutti i processi
    sono campionati con lo stesso seed così le righe sono allineate (la riga i
    del processo A e la riga i del processo B appartengono alla stessa traiettoria).

    Appiattisce tutto in un unico DataFrame dove ogni riga è una traiettoria
    completa: tutte le colonne input e output di ogni processo, più una colonna F.

    Args:
        process_configs: Lista di dizionari configurazione processo (da PROCESSES).
            Ogni elemento deve avere: 'name', 'scm_dataset_type', 'input_labels',
            'output_labels', e per processi ST anche 'st_params' e
            '_st_base_input_labels'/'_st_base_output_labels'.
        n_samples: Numero di campioni da generare per processo.
        seed: Seed per riproducibilità. Tutti i processi usano lo stesso seed.
        noise_mode: 'active' per rumore realistico (default), 'zero' per zero-noise.

    Returns:
        pd.DataFrame con colonne:
            - {process_name}_{input_label} per ogni input di ogni processo
            - {process_name}_{output_label} per ogni output di ogni processo
            - F: reliability calcolata su ogni riga
    """
    all_columns = {}

    for pc in process_configs:
        process_name = pc['name']
        scm_dataset_type = pc['scm_dataset_type']

        # Genera dati SCM per questo processo
        df_process = _sample_process(pc, n_samples, seed, noise_mode)

        # Mappa le colonne SCM alle colonne con prefisso processo.
        # Per processi ST, le label nel config sono già suffissate (es. X_1_p1),
        # ma le colonne nel DataFrame SCM usano le label base (es. X_1).
        input_labels_config = pc['input_labels']
        output_labels_config = pc['output_labels']

        if scm_dataset_type == 'st':
            base_input = pc.get('_st_base_input_labels', input_labels_config)
            base_output = pc.get('_st_base_output_labels', output_labels_config)
        else:
            base_input = input_labels_config
            base_output = output_labels_config

        # Input columns
        for config_label, scm_label in zip(input_labels_config, base_input):
            col_name = f"{process_name}_{config_label}"
            all_columns[col_name] = df_process[scm_label].values

        # Output columns
        for config_label, scm_label in zip(output_labels_config, base_output):
            col_name = f"{process_name}_{config_label}"
            all_columns[col_name] = df_process[scm_label].values

    # Costruisci DataFrame
    df = pd.DataFrame(all_columns)

    # Calcola F per ogni riga
    df['F'] = df.apply(lambda row: compute_F_numpy(row, process_configs), axis=1)

    return df


def _sample_process(
    process_config: dict,
    n_samples: int,
    seed: int,
    noise_mode: str,
) -> pd.DataFrame:
    """
    Campiona dati da un singolo processo SCM.

    Args:
        process_config: Config del processo.
        n_samples: Numero di campioni.
        seed: Random seed.
        noise_mode: 'active' o 'zero'.

    Returns:
        pd.DataFrame con colonne dello SCM (label base, senza prefisso processo).
    """
    scm_dataset_type = process_config['scm_dataset_type']

    if scm_dataset_type == 'st':
        from scm_ds.datasets_st import STConfig, build_st_scm
        st_params = process_config['st_params']
        scm_dataset = build_st_scm(STConfig(**st_params))
    else:
        from scm_ds.datasets import (
            ds_scm_laser, ds_scm_plasma, ds_scm_galvanic, ds_scm_microetch,
            ds_scm_1_to_1_ct,
        )
        _dataset_map = {
            'one_to_one_ct': ds_scm_1_to_1_ct,
            'laser': ds_scm_laser,
            'plasma': ds_scm_plasma,
            'galvanic': ds_scm_galvanic,
            'microetch': ds_scm_microetch,
        }
        if scm_dataset_type not in _dataset_map:
            raise ValueError(f"Unknown SCM dataset type: {scm_dataset_type}")
        scm_dataset = _dataset_map[scm_dataset_type]

    if noise_mode == 'zero':
        # Zero-noise: interviene su tutti i nodi di rumore settandoli a 0
        # Questo è utile per target trajectories ma non per training realistico
        scm_obj = scm_dataset.scm if hasattr(scm_dataset, 'scm') else scm_dataset
        if hasattr(scm_obj, 'do'):
            # Identifica nodi di rumore e intervieni
            noise_interventions = {}
            if hasattr(scm_dataset, 'process_noise_vars'):
                for var in scm_dataset.process_noise_vars:
                    noise_interventions[var] = 0.0
            if noise_interventions:
                scm_obj = scm_obj.do(noise_interventions)
                # Campiona dall'SCM interventato
                return scm_obj.sample(n=n_samples, seed=seed)

    # Campionamento standard con rumore attivo
    return scm_dataset.sample(n=n_samples, seed=seed)


def get_casualit_view(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara la vista globale per CausalIT.

    Prende il DataFrame di traiettorie complete e lo converte nella forma
    (X, y) dove X contiene tutti gli input/output concatenati e y contiene F.

    Returns:
        X: shape (n_samples, total_input_output_dims) — tutti gli inputs e outputs
           concatenati (tutte le colonne tranne F)
        y: shape (n_samples,) — F per ogni riga
    """
    feature_cols = [c for c in df.columns if c != 'F']
    X = df[feature_cols].values
    y = df['F'].values
    return X, y
