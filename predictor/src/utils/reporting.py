from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

def compute_basic_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    return {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std()),
        "max_abs_error": float(abs_errors.max()),
        "p90_abs_error": float(np.percentile(abs_errors, 90)),
        "p95_abs_error": float(np.percentile(abs_errors, 95)),
    }, errors

def save_top_k_worst(y_true, y_pred, output_names, save_csv_path, k=20):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    abs_err = np.abs(y_pred - y_true).sum(axis=1)  # rank by total abs error across outputs
    idx = np.argsort(-abs_err)[:k]
    cols = []
    data = {}
    # build columns for each output
    for i in range(y_true.shape[1]):
        name = output_names[i] if output_names else f"y{i+1}"
        cols += [f"{name}_true", f"{name}_pred", f"{name}_abs_err"]
        data[f"{name}_true"] = y_true[idx, i]
        data[f"{name}_pred"] = y_pred[idx, i]
        data[f"{name}_abs_err"] = np.abs(y_pred[idx, i] - y_true[idx, i])
    df = pd.DataFrame(data, index=idx)
    df.to_csv(save_csv_path, index_label="row_index")
    return save_csv_path

def generate_markdown_report(
    report_dir: Path,
    run_name: str,
    config: dict,
    metrics: dict,
    figures: dict,
    dataset_info: dict,
    residual_stats: dict,
    artifacts: dict,
):
    """
    figures: dict with keys like {"training_history": "path/to/png", "predictions": "...", "error_dist": "..."}
    artifacts: dict with optional paths, e.g. {"checkpoint": "...", "top_worst_csv": "..."}
    dataset_info: {"n_train": int, "n_val": int, "n_test": int, "input_cols": [...], "output_cols": [...]}
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_name}_report.md"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg_json = json.dumps(config, indent=2, ensure_ascii=False)

    # Markdown body
    md = []
    md.append(f"# Experiment Report — {run_name}")
    md.append(f"_Generated: {ts}_\n")

    md.append("## 1. Run Summary")
    md.append(f"- **Model/Run name:** `{run_name}`")
    md.append(f"- **Train/Val/Test sizes:** {dataset_info.get('n_train','?')} / {dataset_info.get('n_val','?')} / {dataset_info.get('n_test','?')}")
    md.append(f"- **Targets:** {', '.join(dataset_info.get('output_cols', [])) or '-'}")
    md.append(f"- **Features (first 10):** {', '.join(dataset_info.get('input_cols', [])[:10]) or '-'}\n")

    md.append("## 2. Metrics")
    # Render metrics as a simple table
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    for k, v in metrics.items():
        if isinstance(v, float):
            md.append(f"| {k} | {v:.6f} |")
        else:
            md.append(f"| {k} | {v} |")
    md.append("")

    if figures.get("training_history"):
        md.append("## 3. Training Curves")
        md.append(f"![training_history]({figures['training_history']})\n")

    if figures.get("predictions"):
        md.append("## 4. Predictions vs Ground Truth")
        md.append(f"![predictions]({figures['predictions']})\n")

    if figures.get("error_dist"):
        md.append("## 5. Error Distribution")
        md.append(f"![error_distribution]({figures['error_dist']})\n")

    md.append("## 6. Residuals Summary")
    md.append("| Stat | Value |")
    md.append("|---|---:|")
    for k, v in residual_stats.items():
        md.append(f"| {k} | {v:.6f} |")
    md.append("")

    if artifacts.get("top_worst_csv"):
        md.append(f"- **Top worst errors CSV:** `{artifacts['top_worst_csv']}`")

    if artifacts.get("checkpoint"):
        md.append(f"- **Checkpoint:** `{artifacts['checkpoint']}`")

    md.append("\n## 7. Config Snapshot")
    md.append("```json")
    md.append(cfg_json)
    md.append("```")

    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path
