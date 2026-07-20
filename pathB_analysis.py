# ============================================================
# FYP2 - PATH B ANALYSIS (B1 + B3)
# Reuses saved predictions_seed*.json — NO retraining needed.
#
# B1: Early-detection latency — samples-to-first-alarm after attack onset
# B3: Early-window evaluation — metrics restricted to first N samples
#     after each attack starts (where the task is genuinely hard)
# ============================================================

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
# Folder where predictions_seed*.json were saved
PRED_DIR = "/content/drive/MyDrive/dataset/processed"
OUT_DIR = "/content/drive/MyDrive/dataset/processed/pathB_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Sampling rate (for converting samples -> seconds)
SAMPLE_RATE_HZ = 2.5

# B3: how many samples after onset to evaluate ("early window")
EARLY_WINDOWS = [5, 10, 20]   # 5 samples = 2s, 10 = 4s, 20 = 8s at 2.5 Hz

# A model is considered to have "alarmed" once it predicts 1 and stays
# consistent. We use a simple persistence rule: first index where the
# model predicts 1 for >= PERSISTENCE consecutive samples.
PERSISTENCE = 2  # require 2 consecutive positives to count as a real alarm


# ============================================================
# LOAD PREDICTIONS
# ============================================================
def load_all_predictions(pred_dir):
    """Load every predictions_seed*.json into a nested dict.

    Returns: {seed: {model_name: {y_true, y_pred, y_score, trajectories, flight_ids}}}
    """
    files = sorted(glob.glob(os.path.join(pred_dir, "predictions_seed*.json")))
    if not files:
        raise FileNotFoundError(
            f"No predictions_seed*.json found in {pred_dir}. "
            "Run the training script first (it saves these per seed)."
        )
    all_preds = {}
    for f in files:
        seed = os.path.basename(f).replace("predictions_seed", "").replace(".json", "")
        with open(f) as fh:
            all_preds[seed] = json.load(fh)
        print(f"Loaded {f} -> models: {list(all_preds[seed].keys())}")
    return all_preds


# ============================================================
# REBUILD PER-FLIGHT TIME SERIES
# ============================================================
def group_by_flight(model_data):
    """Reconstruct per-flight ordered arrays from flattened predictions.

    Within a flight, samples are already in time order (sequences were
    built chronologically), so we keep their original order.
    """
    y_true = np.array(model_data["y_true"])
    y_pred = np.array(model_data["y_pred"])
    y_score = np.array(model_data["y_score"])
    fids = np.array(model_data["flight_ids"])
    trajs = np.array(model_data["trajectories"])

    flights = {}
    for fid in pd.unique(fids):   # pd.unique preserves first-seen order
        mask = (fids == fid)
        flights[fid] = {
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
            "y_score": y_score[mask],
            "trajectory": trajs[mask][0],
        }
    return flights


def find_attack_onset(y_true):
    """Index of the first spoofed sample in a flight. None if no attack."""
    pos = np.where(y_true == 1)[0]
    return int(pos[0]) if len(pos) > 0 else None


def find_first_alarm(y_pred, start_idx, persistence=PERSISTENCE):
    """Index of first sustained alarm at or after start_idx.

    Returns offset (samples after onset) of first alarm, or None if never.
    """
    n = len(y_pred)
    run = 0
    for i in range(start_idx, n):
        if y_pred[i] == 1:
            run += 1
            if run >= persistence:
                # alarm 'fired' at the first sample of this run
                return (i - persistence + 1) - start_idx
        else:
            run = 0
    return None


# ============================================================
# B1: EARLY-DETECTION LATENCY
# ============================================================
def compute_detection_latency(all_preds):
    """For each (seed, model, flight): samples from onset to first alarm."""
    rows = []
    for seed, models in all_preds.items():
        for model_name, model_data in models.items():
            flights = group_by_flight(model_data)
            for fid, fl in flights.items():
                onset = find_attack_onset(fl["y_true"])
                if onset is None:
                    continue  # flight had no attack (shouldn't happen for spoof_*)
                delay = find_first_alarm(fl["y_pred"], onset)
                rows.append({
                    "seed": seed,
                    "model": model_name,
                    "flight_id": fid,
                    "trajectory": fl["trajectory"],
                    "onset_idx": onset,
                    "detection_delay_samples": delay if delay is not None else np.nan,
                    "detection_delay_sec": (delay / SAMPLE_RATE_HZ) if delay is not None else np.nan,
                    "detected": delay is not None,
                })
    return pd.DataFrame(rows)


# ============================================================
# B3: EARLY-WINDOW EVALUATION
# ============================================================
def early_window_metrics(all_preds, window_n):
    """Recompute precision/recall/F1 using only samples in the early window.

    For each spoofed flight we take:
      - the window_n samples immediately AFTER onset (the hard positives)
      - the window_n samples immediately BEFORE onset (clean negatives)
    This focuses evaluation on the moment of attack, where accumulated
    drift has NOT yet made the task trivial.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    rows = []
    for seed, models in all_preds.items():
        for model_name, model_data in models.items():
            flights = group_by_flight(model_data)
            yt_all, yp_all = [], []
            for fid, fl in flights.items():
                onset = find_attack_onset(fl["y_true"])
                if onset is None:
                    continue
                # window after onset (positives)
                post_end = min(onset + window_n, len(fl["y_true"]))
                # window before onset (negatives)
                pre_start = max(onset - window_n, 0)

                yt_all.extend(fl["y_true"][pre_start:post_end])
                yp_all.extend(fl["y_pred"][pre_start:post_end])

            yt_all = np.array(yt_all)
            yp_all = np.array(yp_all)
            if len(yt_all) == 0 or len(np.unique(yt_all)) < 2:
                continue

            rows.append({
                "seed": seed,
                "model": model_name,
                "window_samples": window_n,
                "window_sec": window_n / SAMPLE_RATE_HZ,
                "precision": precision_score(yt_all, yp_all, zero_division=0),
                "recall": recall_score(yt_all, yp_all, zero_division=0),
                "f1": f1_score(yt_all, yp_all, zero_division=0),
                "n_samples": len(yt_all),
            })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("PATH B ANALYSIS — early detection (B1) + early-window eval (B3)")
    print("="*70)

    all_preds = load_all_predictions(PRED_DIR)

    # ---------------- B1 ----------------
    print("\n" + "="*70)
    print("B1: DETECTION LATENCY (samples from attack onset to first alarm)")
    print("="*70)
    lat_df = compute_detection_latency(all_preds)
    lat_df.to_csv(os.path.join(OUT_DIR, "detection_latency_raw.csv"), index=False)

    # Aggregate per model
    print("\nPer-model detection latency (across all seeds & flights):")
    agg = lat_df.groupby("model").agg(
        mean_delay_samples=("detection_delay_samples", "mean"),
        median_delay_samples=("detection_delay_samples", "median"),
        std_delay_samples=("detection_delay_samples", "std"),
        mean_delay_sec=("detection_delay_sec", "mean"),
        detection_rate=("detected", "mean"),
        n_flights=("flight_id", "count"),
    ).reset_index()
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(OUT_DIR, "detection_latency_summary.csv"), index=False)

    # Per-trajectory breakdown
    print("\nPer-model x per-trajectory mean detection delay (samples):")
    pivot = lat_df.pivot_table(
        values="detection_delay_samples",
        index="trajectory", columns="model", aggfunc="mean"
    )
    print(pivot.to_string())

    # ---------------- B3 ----------------
    print("\n" + "="*70)
    print("B3: EARLY-WINDOW EVALUATION (metrics near the moment of attack)")
    print("="*70)
    all_ew = []
    for w in EARLY_WINDOWS:
        ew = early_window_metrics(all_preds, w)
        all_ew.append(ew)
    ew_df = pd.concat(all_ew, ignore_index=True)
    ew_df.to_csv(os.path.join(OUT_DIR, "early_window_raw.csv"), index=False)

    # Aggregate across seeds
    print("\nEarly-window F1 (mean ± std across seeds):")
    ew_agg = ew_df.groupby(["model", "window_samples", "window_sec"]).agg(
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).reset_index().sort_values(["window_samples", "model"])
    print(ew_agg.to_string(index=False))
    ew_agg.to_csv(os.path.join(OUT_DIR, "early_window_summary.csv"), index=False)

    # ---------------- PLOTS ----------------
    # Plot 1: detection latency distribution (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    models = lat_df["model"].unique()
    data = [lat_df[lat_df["model"] == m]["detection_delay_samples"].dropna() for m in models]
    tick_labels = [m.split("(")[0].strip() for m in models]
    try:
        # Matplotlib >= 3.9 renamed the parameter
        ax.boxplot(data, tick_labels=tick_labels)
    except TypeError:
        ax.boxplot(data, labels=tick_labels)
    ax.set_ylabel("Detection delay (samples after onset)")
    ax.set_title("B1: Detection Latency by Model\n(lower = faster detection)")
    ax.grid(alpha=0.3, axis="y")
    # secondary axis in seconds
    secax = ax.secondary_yaxis('right', functions=(lambda x: x / SAMPLE_RATE_HZ,
                                                     lambda x: x * SAMPLE_RATE_HZ))
    secax.set_ylabel("Detection delay (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "B1_detection_latency.png"), dpi=150)
    print(f"\nSaved plot: {OUT_DIR}/B1_detection_latency.png")

    # Plot 2: early-window F1 vs window size
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ew_agg["model"].unique():
        sub = ew_agg[ew_agg["model"] == m]
        ax.errorbar(sub["window_sec"], sub["f1_mean"], yerr=sub["f1_std"],
                    marker="o", capsize=4, label=m.split("(")[0].strip())
    ax.set_xlabel("Early-window size (seconds after onset)")
    ax.set_ylabel("F1 score")
    ax.set_title("B3: Early-Window F1\n(performance right after attack starts)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "B3_early_window_f1.png"), dpi=150)
    print(f"Saved plot: {OUT_DIR}/B3_early_window_f1.png")

    print("\n" + "="*70)
    print("DONE. Key files in", OUT_DIR)
    print("  detection_latency_summary.csv  <- B1 headline numbers")
    print("  early_window_summary.csv       <- B3 headline numbers")
    print("  B1_detection_latency.png, B3_early_window_f1.png")
    print("="*70)


if __name__ == "__main__":
    main()
