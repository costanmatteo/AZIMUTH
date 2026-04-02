"""
Converter da formato traiettorie AZIMUTH a formato causaliT.

Legge full_trajectories.pt e produce:
  - ds.npz          (s, x, y arrays con shape [n_campioni, n_vars, 2])
  - dataset_metadata.json
  - Maschere di attenzione CSV (enc_self_att_mask, dec_self_att_mask, dec_cross_att_mask,
    dag_adj_mask, dec1_cross_att_mask, dec1_self_att_mask, dec2_cross_att_mask, dec2_self_att_mask)

Uso:
    python scm_ds/convert_azimuth_trajectories.py \
        --input scm_ds/predictor_dataset/trajectories/full_trajectories.pt \
        --output scm_ds/causalit_dataset/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_trajectories(input_path: str):
    """Carica le traiettorie AZIMUTH da file .pt."""
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    if not data:
        raise ValueError(f"File vuoto: {input_path}")
    return data


def extract_process_info(trajectories):
    """Estrae la lista ordinata dei processi e le dimensioni dai dati."""
    sample = trajectories[0]["trajectory"]
    process_names = list(sample.keys())

    process_dims = {}
    for pname, pdata in sample.items():
        process_dims[pname] = {
            "n_inputs": pdata["inputs"].shape[0],
            "n_env": pdata["env"].shape[0],
            "n_outputs": pdata["outputs"].shape[0],
        }
    return process_names, process_dims


def build_arrays(trajectories, process_names, process_dims):
    """Costruisce gli array S, X, Y con variable IDs per causaliT.

    Restituisce:
        s_array: (n_campioni, n_source, 2)  — [valore, id_variabile] (id 1-based, locale)
        x_array: (n_campioni, n_input, 2)
        y_array: (n_campioni, n_target, 2)
        source_labels: lista di nomi variabili S
        input_labels: lista di nomi variabili X (output dei processi)
        target_labels: lista di nomi variabili Y (F)
    """
    n_campioni = len(trajectories)

    # Calcola dimensioni totali
    n_source = sum(d["n_inputs"] + d["n_env"] for d in process_dims.values())
    n_input = sum(d["n_outputs"] for d in process_dims.values())
    n_target = 1  # F

    s_array = np.zeros((n_campioni, n_source, 2), dtype=np.float32)
    x_array = np.zeros((n_campioni, n_input, 2), dtype=np.float32)
    y_array = np.zeros((n_campioni, n_target, 2), dtype=np.float32)

    # Costruisci label e assegna ID (1-based, locali per gruppo)
    source_labels = []
    for pname in process_names:
        dims = process_dims[pname]
        for i in range(1, dims["n_inputs"] + 1):
            source_labels.append(f"X_{i}_{pname}")
        for i in range(1, dims["n_env"] + 1):
            source_labels.append(f"E_{i}_{pname}")

    input_labels = []
    for pname in process_names:
        dims = process_dims[pname]
        for i in range(1, dims["n_outputs"] + 1):
            input_labels.append(f"Y_{i}_{pname}" if dims["n_outputs"] > 1 else f"Y_{pname}")

    target_labels = ["F"]

    # ID locali 1-based per ogni gruppo
    s_ids = np.arange(1, n_source + 1, dtype=np.float32)
    x_ids = np.arange(1, n_input + 1, dtype=np.float32)
    y_ids = np.array([1.0], dtype=np.float32)

    # Riempi gli array
    for idx, traj_dict in enumerate(trajectories):
        traj = traj_dict["trajectory"]
        F_val = traj_dict["F"]

        # S: concatena inputs + env di ogni processo
        s_col = 0
        for pname in process_names:
            pdata = traj[pname]
            inputs = pdata["inputs"].numpy().flatten()
            env = pdata["env"].numpy().flatten()
            vals = np.concatenate([inputs, env])
            for v in vals:
                s_array[idx, s_col, 0] = v
                s_array[idx, s_col, 1] = s_ids[s_col]
                s_col += 1

        # X: concatena outputs di ogni processo
        x_col = 0
        for pname in process_names:
            pdata = traj[pname]
            outputs = pdata["outputs"].numpy().flatten()
            for v in outputs:
                x_array[idx, x_col, 0] = v
                x_array[idx, x_col, 1] = x_ids[x_col]
                x_col += 1

        # Y: F
        y_array[idx, 0, 0] = float(F_val)
        y_array[idx, 0, 1] = y_ids[0]

    return s_array, x_array, y_array, source_labels, input_labels, target_labels


def build_masks(process_names, process_dims, source_labels, input_labels, target_labels):
    """Costruisce le maschere di attenzione causale.

    Restituisce un dizionario {nome_file: DataFrame}.
    """
    n_source = len(source_labels)
    n_input = len(input_labels)
    n_target = len(target_labels)

    # --- enc_self_att_mask (S↔S): blocchi diagonali per processo ---
    enc_self = np.zeros((n_source, n_source), dtype=int)
    s_offset = 0
    for pname in process_names:
        dims = process_dims[pname]
        block_size = dims["n_inputs"] + dims["n_env"]
        enc_self[s_offset:s_offset + block_size, s_offset:s_offset + block_size] = 1
        s_offset += block_size

    # --- dec_self_att_mask (X↔X): triangolare inferiore (catena causale) ---
    dec_self = np.tril(np.ones((n_input, n_input), dtype=int))

    # --- dec_cross_att_mask (S→X): blocchi, output_i guarda solo le S del processo i ---
    dec_cross = np.zeros((n_input, n_source), dtype=int)
    x_offset = 0
    s_offset = 0
    for pname in process_names:
        dims = process_dims[pname]
        s_block = dims["n_inputs"] + dims["n_env"]
        o_block = dims["n_outputs"]
        dec_cross[x_offset:x_offset + o_block, s_offset:s_offset + s_block] = 1
        x_offset += o_block
        s_offset += s_block

    # --- Maschere aggiuntive per StageCausaliT (dec1/dec2 naming) ---
    # dec1_cross = dec_cross (X attends to S)
    # dec1_self = dec_self (X attends to X)
    # dec2_cross (Y attends to X): Y può guardare tutti gli X
    dec2_cross = np.ones((n_target, n_input), dtype=int)
    # dec2_self (Y attends to Y): singola variabile, sempre 1
    dec2_self = np.ones((n_target, n_target), dtype=int)

    # --- dag_adj_mask: matrice completa di adiacenza ---
    all_labels = source_labels + input_labels + target_labels
    n_all = len(all_labels)
    dag_adj = np.zeros((n_all, n_all), dtype=int)
    # S↔S block
    dag_adj[:n_source, :n_source] = enc_self
    # S→X block
    dag_adj[n_source:n_source + n_input, :n_source] = dec_cross
    # X↔X block
    dag_adj[n_source:n_source + n_input, n_source:n_source + n_input] = dec_self
    # X→Y block
    dag_adj[n_source + n_input:, n_source:n_source + n_input] = dec2_cross
    # Y↔Y block
    dag_adj[n_source + n_input:, n_source + n_input:] = dec2_self

    masks = {
        # 3 maschere richieste dal task (formato no-source)
        "enc_self_att_mask.csv": pd.DataFrame(enc_self, index=source_labels, columns=source_labels),
        "dec_self_att_mask.csv": pd.DataFrame(dec_self, index=input_labels, columns=input_labels),
        "dec_cross_att_mask.csv": pd.DataFrame(dec_cross, index=input_labels, columns=source_labels),
        # Maschere StageCausaliT (formato source-present)
        "dag_adj_mask.csv": pd.DataFrame(dag_adj, index=all_labels, columns=all_labels),
        "dec1_cross_att_mask.csv": pd.DataFrame(dec_cross, index=input_labels, columns=source_labels),
        "dec1_self_att_mask.csv": pd.DataFrame(dec_self, index=input_labels, columns=input_labels),
        "dec2_cross_att_mask.csv": pd.DataFrame(dec2_cross, index=target_labels, columns=input_labels),
        "dec2_self_att_mask.csv": pd.DataFrame(dec2_self, index=target_labels, columns=target_labels),
    }
    return masks


def build_metadata(source_labels, input_labels, target_labels):
    """Costruisce il dataset_metadata.json."""
    return {
        "variable_info": {
            "source_labels": source_labels,
            "input_labels": input_labels,
            "target_labels": target_labels,
            "n_source": len(source_labels),
            "n_input": len(input_labels),
            "n_target": len(target_labels),
        },
        "feature_indices": {
            "value": 0,
            "variable": 1,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Converte traiettorie AZIMUTH in formato causaliT"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="scm_ds/predictor_dataset/trajectories/full_trajectories.pt",
        help="Path al file full_trajectories.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scm_ds/causalit_dataset",
        help="Directory di output per il dataset causaliT",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AZIMUTH → causaliT Converter")
    print("=" * 60)

    # 1. Carica traiettorie
    print(f"\n[1/4] Caricamento traiettorie da {input_path}...")
    trajectories = load_trajectories(str(input_path))
    print(f"  Campioni: {len(trajectories)}")

    # 2. Estrai info processi
    print("\n[2/4] Analisi struttura processi...")
    process_names, process_dims = extract_process_info(trajectories)
    for pname in process_names:
        d = process_dims[pname]
        print(f"  {pname}: {d['n_inputs']} inputs, {d['n_env']} env, {d['n_outputs']} outputs")

    # 3. Costruisci array
    print("\n[3/4] Costruzione array S, X, Y...")
    s_array, x_array, y_array, source_labels, input_labels, target_labels = build_arrays(
        trajectories, process_names, process_dims
    )
    print(f"  S (source):  {s_array.shape}  labels: {source_labels}")
    print(f"  X (input):   {x_array.shape}  labels: {input_labels}")
    print(f"  Y (target):  {y_array.shape}  labels: {target_labels}")

    # 4. Salva tutto
    print(f"\n[4/4] Salvataggio in {output_dir}/...")

    # ds.npz
    npz_path = output_dir / "ds.npz"
    np.savez_compressed(str(npz_path), s=s_array, x=x_array, y=y_array)
    print(f"  {npz_path}")

    # dataset_metadata.json
    metadata = build_metadata(source_labels, input_labels, target_labels)
    meta_path = output_dir / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"  {meta_path}")

    # Maschere di attenzione
    masks = build_masks(process_names, process_dims, source_labels, input_labels, target_labels)
    for filename, df in masks.items():
        mask_path = output_dir / filename
        df.to_csv(mask_path)
        print(f"  {mask_path}  shape={df.shape}")

    print("\n" + "=" * 60)
    print("Conversione completata!")
    print("=" * 60)


if __name__ == "__main__":
    main()
