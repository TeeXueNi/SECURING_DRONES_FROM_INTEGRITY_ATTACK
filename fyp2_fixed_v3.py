# ============================================================
# FYP2 IMPROVED COMPARISON CODE (v2 — flight-aware)
# - Hybrid: LSTM motion predictor (regression) -> residuals -> XGBoost -> fusion
# - Ablation: LSTM classifier + XGBoost (snapshot) -> fusion
#
# UPDATES IN v2:
# - Reads new dataset format with flight_id / trajectory / per-timestep labels
# - Sequences built PER FLIGHT (no cross-flight contamination)
# - Train/val/test split by FLIGHT, not by row (tests generalization)
# - Test set includes held-out attack configurations
# - Per-trajectory and per-attack-type result breakdown
# ============================================================

import os
import time
import random
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, roc_curve, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier

# Silence the manylinux2014 XGBoost warning
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Data
    "csv_path": "final_dataset.csv",
    "window": 50,           # 50 samples @ 2.5 Hz = 20 seconds of history
    "horizon": 1,
    "predict_cols": ["lat", "lon", "alt", "vel_n", "vel_e", "vel_d"],
    # Columns that aren't features (and shouldn't be fed to models)
    "non_feature_cols": ["timestamp", "label", "flight_id", "trajectory"],

    # DL training
    "epochs": 15,
    "batch": 64,
    "patience": 4,

    # LSTM architecture
    "lstm_units": 128,
    "dense_units": 64,
    "dropout_rate": 0.3,

    # XGBoost
    "xgb_trees": 700,
    "xgb_depth": 6,
    "xgb_lr": 0.05,
    "xgb_subsample": 0.8,
    "xgb_colsample": 0.8,

    # Fusion search grid
    "w_grid": np.arange(0.0, 1.01, 0.05),
    "t_grid": np.arange(0.10, 0.91, 0.05),

    # Reproducibility
    "seeds": [42, 123, 456],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_sequences_per_flight(df, feature_cols, window, horizon):
    """Build sequence windows PER FLIGHT (no cross-flight contamination).

    Returns:
        X_seq: (N, window, F)        sequence inputs
        X_snap: (N, F)               snapshot inputs (next-step row)
        y_cls: (N,)                  classification labels
        flight_ids: (N,)             flight_id for each sample
        trajectories: (N,)           trajectory for each sample
    """
    X_seq_all, X_snap_all = [], []
    y_cls_all = []
    flight_ids_all, trajs_all = [], []

    for fid, group in df.groupby("flight_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < window + horizon + 1:
            print(f"  Skipping {fid}: only {len(group)} rows (need >{window+horizon})")
            continue

        X_flight = group[feature_cols].values.astype(np.float32)
        y_flight = group["label"].values.astype(np.int32)
        traj = group["trajectory"].iloc[0]

        for t in range(window, len(X_flight) - horizon):
            X_seq_all.append(X_flight[t-window:t])
            X_snap_all.append(X_flight[t+horizon])
            y_cls_all.append(y_flight[t+horizon])
            flight_ids_all.append(fid)
            trajs_all.append(traj)

    return (np.array(X_seq_all, dtype=np.float32),
            np.array(X_snap_all, dtype=np.float32),
            np.array(y_cls_all, dtype=np.int32),
            np.array(flight_ids_all),
            np.array(trajs_all))


def split_by_flight(flight_ids, train_flights, val_flights, test_flights):
    """Return boolean masks for train/val/test based on flight_id."""
    train_mask = np.isin(flight_ids, list(train_flights))
    val_mask = np.isin(flight_ids, list(val_flights))
    test_mask = np.isin(flight_ids, list(test_flights))
    return train_mask, val_mask, test_mask


def assign_flights_to_splits(all_flight_ids, seed=42):
    """Decide which flights go to train/val/test.

    Strategy:
      - ALL normal flights -> train (they're our 'clean physics' substrate)
      - Spoofed flights divided so test contains diverse, held-out configs
      - Test set explicitly includes one of each attack BLOCK so we can
        report per-block metrics
    """
    rng = np.random.default_rng(seed)
    all_flight_ids = list(all_flight_ids)

    # Normal flights -> all to train (LSTM predictor needs lots of clean data)
    normal_flights = [f for f in all_flight_ids if f.startswith("normal_")]
    spoof_flights = [f for f in all_flight_ids if f.startswith("spoof_")]

    # Categorize spoofed flights by block (mag / dir / onset)
    mag_flights = [f for f in spoof_flights if "_mag" in f]
    dir_flights = [f for f in spoof_flights if "_dir" in f]
    onset_flights = [f for f in spoof_flights if "_onset" in f]

    def split_group(group, n_test, n_val):
        """Randomly assign within a group."""
        g = list(group)
        rng.shuffle(g)
        return g[:n_test], g[n_test:n_test+n_val], g[n_test+n_val:]

    # Per-block split: roughly 65% train / 15% val / 20% test of each block
    mag_te, mag_v, mag_tr = split_group(mag_flights,
                                         max(1, len(mag_flights) // 5),
                                         max(1, len(mag_flights) // 7))
    dir_te, dir_v, dir_tr = split_group(dir_flights,
                                         max(1, len(dir_flights) // 5),
                                         max(1, len(dir_flights) // 7))
    onset_te, onset_v, onset_tr = split_group(onset_flights,
                                               max(1, len(onset_flights) // 5),
                                               max(1, len(onset_flights) // 7))

    train_flights = set(normal_flights + mag_tr + dir_tr + onset_tr)
    val_flights = set(mag_v + dir_v + onset_v)
    test_flights = set(mag_te + dir_te + onset_te)

    return train_flights, val_flights, test_flights


def scale_sequences(scaler, X_seq):
    X2 = X_seq.reshape(-1, X_seq.shape[2])
    X2s = scaler.transform(X2)
    return X2s.reshape(X_seq.shape)


def calculate_far(y_true, y_pred):
    """False Alarm Rate = FP / (FP + TN)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def tune_threshold(y_true, y_score, t_grid):
    """Find threshold that maximizes F1 on validation set."""
    best_t, best_f1 = 0.5, -1.0
    for t in t_grid:
        pred = (y_score > t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_model(name, y_true, y_pred, y_score, latency_ms=None):
    """Compute all metrics for a model and return as dict."""
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": float(calculate_far(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else None,
        "latency_ms": latency_ms,
    }
    return metrics


def print_metrics(metrics):
    print(f"\n===== {metrics['model']} =====")
    print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-score:    {metrics['f1']:.4f}")
    print(f"FAR:         {metrics['far']*100:.2f}%")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    if metrics["latency_ms"] is not None:
        print(f"Latency:     {metrics['latency_ms']:.3f} ms/sample")


def breakdown_by_group(y_true, y_pred, groups, group_name="group"):
    """Compute precision/recall/F1 per group (e.g., per trajectory)."""
    rows = []
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        rows.append({
            group_name: g,
            "n_samples": int(mask.sum()),
            "n_positive": int((y_true[mask] == 1).sum()),
            "precision": precision_score(y_true[mask], y_pred[mask], zero_division=0),
            "recall": recall_score(y_true[mask], y_pred[mask], zero_division=0),
            "f1": f1_score(y_true[mask], y_pred[mask], zero_division=0),
            "far": calculate_far(y_true[mask], y_pred[mask]),
        })
    return pd.DataFrame(rows)


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_lstm_predictor(window, n_features, target_dim):
    """LSTM regressor: predicts next-step motion state."""
    return Sequential([
        LSTM(CONFIG["lstm_units"], input_shape=(window, n_features)),
        Dropout(CONFIG["dropout_rate"]),
        Dense(CONFIG["dense_units"], activation="relu"),
        Dropout(0.2),
        Dense(target_dim, activation="linear")
    ])


def build_lstm_classifier(window, n_features):
    """LSTM classifier: directly outputs spoofing probability (for ablation)."""
    return Sequential([
        LSTM(CONFIG["lstm_units"], input_shape=(window, n_features)),
        Dropout(CONFIG["dropout_rate"]),
        Dense(CONFIG["dense_units"], activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])


# ============================================================
# MAIN EXPERIMENT (single seed run)
# ============================================================
def run_single_seed(seed):
    """Run the full pipeline once with a given seed. Returns metrics dict."""
    print(f"\n{'='*70}")
    print(f"RUNNING SEED = {seed}")
    print(f"{'='*70}")
    set_seeds(seed)

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(CONFIG["csv_path"])
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Unique flights: {df['flight_id'].nunique()}")

    # Determine feature columns (everything except non-feature cols)
    feature_cols = [c for c in df.columns if c not in CONFIG["non_feature_cols"]]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ----------------------------
    # Build sequences PER FLIGHT
    # ----------------------------
    print("\nBuilding sequences per flight...")
    X_seq, X_snap, y_cls, fids, trajs = build_sequences_per_flight(
        df, feature_cols, CONFIG["window"], CONFIG["horizon"]
    )
    print(f"Built {len(X_seq)} sequence samples from {len(np.unique(fids))} flights")
    print(f"Overall class balance: {np.bincount(y_cls)}")

    # ----------------------------
    # Flight-level train/val/test split
    # ----------------------------
    all_flight_ids = np.unique(fids)
    train_flights, val_flights, test_flights = assign_flights_to_splits(
        all_flight_ids, seed=seed
    )
    print(f"\nSplit by FLIGHT:")
    print(f"  Train: {len(train_flights)} flights")
    print(f"  Val:   {len(val_flights)} flights -> {sorted(val_flights)}")
    print(f"  Test:  {len(test_flights)} flights -> {sorted(test_flights)}")

    train_mask, val_mask, test_mask = split_by_flight(
        fids, train_flights, val_flights, test_flights
    )

    X_seq_tr = X_seq[train_mask]; X_seq_v = X_seq[val_mask]; X_seq_te = X_seq[test_mask]
    X_snap_tr = X_snap[train_mask]; X_snap_v = X_snap[val_mask]; X_snap_te = X_snap[test_mask]
    y_tr = y_cls[train_mask]; y_v = y_cls[val_mask]; y_te = y_cls[test_mask]
    trajs_te = trajs[test_mask]
    fids_te = fids[test_mask]

    print(f"\nSample counts:")
    print(f"  Train: {len(y_tr)} | class balance: {np.bincount(y_tr)}")
    print(f"  Val:   {len(y_v)} | class balance: {np.bincount(y_v)}")
    print(f"  Test:  {len(y_te)} | class balance: {np.bincount(y_te)}")

    # Sanity check: if val or test has no positives, results will be unreliable
    if len(y_v) == 0 or len(y_te) == 0:
        print("\n!!! WARNING: val or test set is empty. Skipping seed.")
        return [], {}
    if (y_v == 1).sum() == 0:
        print("\n!!! WARNING: val set has no positive samples. Threshold tuning won't work well.")
    if (y_te == 1).sum() == 0:
        print("\n!!! WARNING: test set has no positive samples.")

    # ----------------------------
    # Scaling
    # ----------------------------
    seq_scaler = StandardScaler()
    seq_scaler.fit(X_seq_tr.reshape(-1, X_seq_tr.shape[2]))
    X_seq_tr_s = scale_sequences(seq_scaler, X_seq_tr)
    X_seq_v_s = scale_sequences(seq_scaler, X_seq_v)
    X_seq_te_s = scale_sequences(seq_scaler, X_seq_te)

    snap_scaler = StandardScaler()
    snap_scaler.fit(X_snap_tr)
    X_snap_tr_s = snap_scaler.transform(X_snap_tr)
    X_snap_v_s = snap_scaler.transform(X_snap_v)
    X_snap_te_s = snap_scaler.transform(X_snap_te)

    # Class weights for DL models
    if len(np.unique(y_tr)) > 1:
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        class_weight_dict = dict(enumerate(cw))
    else:
        class_weight_dict = None
    print(f"\nClass weights: {class_weight_dict}")

    F = X_seq_tr_s.shape[2]
    all_metrics = []
    roc_data = {}
    saved_predictions = {}  # for later analysis

    # ============================================================
    # PART C: HYBRID MODEL (LSTM predictor + XGBoost + fusion)
    # ============================================================
    print("\n----- PART C: HYBRID MODEL -----")

    pred_idx = [feature_cols.index(c) for c in CONFIG["predict_cols"]]
    target_dim = len(pred_idx)

    # Build next-step regression targets aligned with X_snap
    y_next = X_snap[:, pred_idx]  # what the LSTM tries to predict
    y_next_tr = y_next[train_mask]
    y_next_v = y_next[val_mask]
    y_next_te = y_next[test_mask]
    cur_te_raw = X_snap[test_mask]  # raw (un-scaled) snapshot for residual features

    # Scale the regression targets the same way as sequence features (they ARE features)
    # We'll use the seq scaler's stats on just the predict_idx columns
    # Easier: use a fresh scaler that matches what the LSTM sees
    target_scaler = StandardScaler()
    target_scaler.fit(y_next_tr)
    y_next_tr_s = target_scaler.transform(y_next_tr)
    y_next_v_s = target_scaler.transform(y_next_v)
    y_next_te_s = target_scaler.transform(y_next_te)

    # Train LSTM predictor on NORMAL samples only
    lstm_pred = build_lstm_predictor(CONFIG["window"], F, target_dim)
    lstm_pred.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=CONFIG["patience"],
                       restore_best_weights=True)

    normal_mask_tr = (y_tr == 0)
    normal_mask_v = (y_v == 0)
    X_tr_norm = X_seq_tr_s[normal_mask_tr]
    y_tr_norm = y_next_tr_s[normal_mask_tr]

    if np.any(normal_mask_v):
        X_v_norm = X_seq_v_s[normal_mask_v]
        y_v_norm = y_next_v_s[normal_mask_v]
    else:
        # Fall back to using all val (shouldn't happen with our split)
        X_v_norm, y_v_norm = X_seq_v_s, y_next_v_s

    print(f"LSTM predictor: training on {len(X_tr_norm)} normal samples")
    lstm_pred.fit(
        X_tr_norm, y_tr_norm,
        validation_data=(X_v_norm, y_v_norm),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch"],
        callbacks=[es],
        verbose=1
    )
    lstm_pred.save(f"hybrid_predictor_seed{seed}.keras")

    def build_residual_features(X_seq_s, y_next_true_s, cur_raw):
        """Build XGBoost feature vector from prediction residuals + raw snapshot."""
        y_hat = lstm_pred.predict(X_seq_s, verbose=0)
        residual = y_next_true_s - y_hat
        abs_res = np.abs(residual)
        anomaly = np.mean(residual**2, axis=1, keepdims=True)
        feats = np.concatenate([cur_raw, residual, abs_res, anomaly], axis=1)
        return anomaly, feats

    anom_tr, X_xgb_tr = build_residual_features(X_seq_tr_s, y_next_tr_s, X_snap_tr_s)
    anom_v, X_xgb_v = build_residual_features(X_seq_v_s, y_next_v_s, X_snap_v_s)
    anom_te, X_xgb_te = build_residual_features(X_seq_te_s, y_next_te_s, X_snap_te_s)

    # XGBoost classifier
    pos = np.sum(y_tr == 1)
    neg = np.sum(y_tr == 0)
    spw = (neg / pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=CONFIG["xgb_trees"],
        max_depth=CONFIG["xgb_depth"],
        learning_rate=CONFIG["xgb_lr"],
        subsample=CONFIG["xgb_subsample"],
        colsample_bytree=CONFIG["xgb_colsample"],
        reg_lambda=1.0,
        min_child_weight=1,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=spw,
    )
    xgb.fit(X_xgb_tr, y_tr)
    joblib.dump(xgb, f"hybrid_xgb_seed{seed}.pkl")

    # Fusion scores
    anom_scaler = MinMaxScaler()
    anom_scaler.fit(anom_tr)
    lstm_score_v = anom_scaler.transform(anom_v).flatten()
    lstm_score_te = anom_scaler.transform(anom_te).flatten()

    xgb_prob_v = xgb.predict_proba(X_xgb_v)[:, 1]
    xgb_prob_te = xgb.predict_proba(X_xgb_te)[:, 1]

    # Tune both w and t on VAL
    best = {"w": 0.5, "t": 0.5, "f1": -1.0}
    for w in CONFIG["w_grid"]:
        fused = w * lstm_score_v + (1 - w) * xgb_prob_v
        for t in CONFIG["t_grid"]:
            pred = (fused > t).astype(int)
            f1 = f1_score(y_v, pred, zero_division=0)
            if f1 > best["f1"]:
                best = {"w": float(w), "t": float(t), "f1": float(f1)}
    print(f"\nBest fusion params (on val): w={best['w']:.2f}, t={best['t']:.2f}, F1={best['f1']:.4f}")

    fused_te = best["w"] * lstm_score_te + (1 - best["w"]) * xgb_prob_te
    hybrid_pred = (fused_te > best["t"]).astype(int)

    # Measure hybrid latency (FIX 1: direct model call, not .predict();
    # FIX 3: time XGBoost separately since it carries the detection signal)
    one_seq = X_seq_te_s[:1]
    one_xgb = X_xgb_te[:1]

    # Warm up the direct call once (triggers any graph build)
    _ = lstm_pred(one_seq, training=False)

    # Full pipeline: LSTM (direct call) + XGBoost
    full_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = lstm_pred(one_seq, training=False).numpy()
        _ = xgb.predict_proba(one_xgb)
        full_times.append((time.perf_counter() - start) * 1000)
    hybrid_lat = float(np.mean(full_times))

    # XGBoost-only latency (the actual detector, comparable to tree-based papers)
    xgb_times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = xgb.predict_proba(one_xgb)
        xgb_times.append((time.perf_counter() - start) * 1000)
    hybrid_xgb_lat = float(np.mean(xgb_times))
    print(f"\nHybrid latency: full pipeline {hybrid_lat:.3f} ms/sample, "
          f"XGBoost-only {hybrid_xgb_lat:.3f} ms/sample")

    m = evaluate_model("HYBRID (Ours)", y_te, hybrid_pred, fused_te, hybrid_lat)
    m["xgb_only_latency_ms"] = hybrid_xgb_lat
    m["threshold"] = best["t"]
    m["alpha"] = best["w"]
    print_metrics(m)
    all_metrics.append(m)
    roc_data["HYBRID"] = (y_te, fused_te)
    saved_predictions["HYBRID"] = {
        "y_true": y_te.tolist(),
        "y_pred": hybrid_pred.tolist(),
        "y_score": fused_te.tolist(),
        "trajectories": trajs_te.tolist(),
        "flight_ids": fids_te.tolist(),
    }

    # Per-trajectory breakdown
    print("\n  Per-trajectory breakdown (HYBRID):")
    print(breakdown_by_group(y_te, hybrid_pred, trajs_te, "trajectory").to_string(index=False))

    # ============================================================
    # PART D: ABLATION - LSTM as classifier + XGBoost fusion
    # ============================================================
    print("\n----- PART D: ABLATION (LSTM-classifier + XGBoost) -----")

    lstm_cls_model = build_lstm_classifier(CONFIG["window"], F)
    lstm_cls_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    es_cls = EarlyStopping(monitor="val_loss", patience=CONFIG["patience"],
                           restore_best_weights=True)
    lstm_cls_model.fit(
        X_seq_tr_s, y_tr,
        validation_data=(X_seq_v_s, y_v),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch"],
        callbacks=[es_cls],
        class_weight=class_weight_dict,
        verbose=1
    )
    lstm_cls_model.save(f"ablation_lstm_cls_seed{seed}.keras")

    lstm_cls_v = lstm_cls_model.predict(X_seq_v_s, verbose=0).flatten()
    lstm_cls_te = lstm_cls_model.predict(X_seq_te_s, verbose=0).flatten()

    # XGBoost on snapshot features
    xgb_snap = XGBClassifier(
        n_estimators=CONFIG["xgb_trees"], max_depth=CONFIG["xgb_depth"],
        learning_rate=CONFIG["xgb_lr"], random_state=seed,
        scale_pos_weight=spw, eval_metric="logloss"
    )
    xgb_snap.fit(X_snap_tr_s, y_tr)
    xgb_snap_v = xgb_snap.predict_proba(X_snap_v_s)[:, 1]
    xgb_snap_te = xgb_snap.predict_proba(X_snap_te_s)[:, 1]

    # Tune fusion
    best_abl = {"w": 0.5, "t": 0.5, "f1": -1.0}
    for w in CONFIG["w_grid"]:
        fused = w * lstm_cls_v + (1 - w) * xgb_snap_v
        for t in CONFIG["t_grid"]:
            pred = (fused > t).astype(int)
            f1 = f1_score(y_v, pred, zero_division=0)
            if f1 > best_abl["f1"]:
                best_abl = {"w": float(w), "t": float(t), "f1": float(f1)}
    print(f"\nBest ablation fusion params: w={best_abl['w']:.2f}, t={best_abl['t']:.2f}, F1={best_abl['f1']:.4f}")

    fused_abl_te = best_abl["w"] * lstm_cls_te + (1 - best_abl["w"]) * xgb_snap_te
    abl_pred = (fused_abl_te > best_abl["t"]).astype(int)

    # Latency for ablation (same method as hybrid, for a fair comparison)
    one_seq_a = X_seq_te_s[:1]
    one_snap_a = X_snap_te_s[:1]

    _ = lstm_cls_model(one_seq_a, training=False)  # warm up

    full_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = lstm_cls_model(one_seq_a, training=False).numpy()
        _ = xgb_snap.predict_proba(one_snap_a)
        full_times.append((time.perf_counter() - start) * 1000)
    abl_lat = float(np.mean(full_times))

    xgb_times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = xgb_snap.predict_proba(one_snap_a)
        xgb_times.append((time.perf_counter() - start) * 1000)
    abl_xgb_lat = float(np.mean(xgb_times))
    print(f"\nAblation latency: full pipeline {abl_lat:.3f} ms/sample, "
          f"XGBoost-only {abl_xgb_lat:.3f} ms/sample")

    m = evaluate_model("ABLATION (LSTM-cls + XGB)", y_te, abl_pred, fused_abl_te, abl_lat)
    m["xgb_only_latency_ms"] = abl_xgb_lat
    m["threshold"] = best_abl["t"]
    m["alpha"] = best_abl["w"]
    print_metrics(m)
    all_metrics.append(m)
    roc_data["ABLATION"] = (y_te, fused_abl_te)
    saved_predictions["ABLATION"] = {
        "y_true": y_te.tolist(),
        "y_pred": abl_pred.tolist(),
        "y_score": fused_abl_te.tolist(),
        "trajectories": trajs_te.tolist(),
        "flight_ids": fids_te.tolist(),
    }

    print("\n  Per-trajectory breakdown (ABLATION):")
    print(breakdown_by_group(y_te, abl_pred, trajs_te, "trajectory").to_string(index=False))

    # Save predictions for this seed
    with open(f"predictions_seed{seed}.json", "w") as f:
        json.dump(saved_predictions, f, indent=2)

    return all_metrics, roc_data


# ============================================================
# MULTI-SEED EXPERIMENT
# ============================================================
def main():
    all_runs = []
    last_roc = None

    for seed in CONFIG["seeds"]:
        metrics, roc_data = run_single_seed(seed)
        if metrics:
            all_runs.append(metrics)
            last_roc = roc_data

    if not all_runs:
        print("\n!!! No successful runs. Exiting.")
        return

    # ----------------------------
    # Aggregate across seeds
    # ----------------------------
    print(f"\n\n{'='*70}")
    print(f"AGGREGATED RESULTS ACROSS {len(all_runs)} SEEDS")
    print(f"{'='*70}")

    by_model = {}
    for run in all_runs:
        for m in run:
            by_model.setdefault(m["model"], []).append(m)

    summary_rows = []
    for model_name, runs in by_model.items():
        row = {"model": model_name}
        for key in ["accuracy", "precision", "recall", "f1", "far", "roc_auc", "latency_ms"]:
            vals = [r[key] for r in runs if r[key] is not None]
            if vals:
                row[f"{key}_mean"] = np.mean(vals)
                row[f"{key}_std"] = np.std(vals)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n", summary_df.to_string(index=False))
    summary_df.to_csv("results_summary.csv", index=False)
    print("\nSummary saved to results_summary.csv")

    with open("results_full.json", "w") as f:
        json.dump(all_runs, f, indent=2, default=str)
    print("Full results saved to results_full.json")

    # ----------------------------
    # Plot ROC curves
    # ----------------------------
    if last_roc:
        plt.figure(figsize=(10, 8))
        for name, (y_true, y_score) in last_roc.items():
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - GPS Spoofing Detection")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("roc_curves.png", dpi=150)
        print("ROC curves saved to roc_curves.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
