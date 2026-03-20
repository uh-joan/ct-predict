#!/usr/bin/env python3
"""
Clinical Trial Outcome Predictor — train.py

This is the ONLY file the autoresearch agent modifies.
Everything is here: feature engineering, model definition, training.

Run:  python train.py > run.log 2>&1
Eval: python prepare.py  (extracts auc_roc from grep-friendly output)
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (do not change)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# ---------------------------------------------------------------------------
# Feature engineering — MODIFY THIS
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y from raw trial data.
    """
    y = df["label"].astype(int)

    # Numeric features
    numeric_cols = [
        "phase", "enrollment", "num_arms", "has_dmc",
        "num_secondary_endpoints", "num_sites", "has_biomarker_selection",
        "competitor_trial_count", "prior_phase_success",
        # OpenTargets
        "ot_overall_score", "ot_target_tractability", "target_disease_score",
        "ot_genetic_score", "ot_somatic_score", "ot_literature_score",
        "ot_animal_model_score", "ot_known_drug_score", "ot_affected_pathway_score",
        # ChEMBL
        "chembl_selectivity", "chembl_best_ic50_nm", "chembl_num_assays",
        "chembl_max_phase", "chembl_moa_count",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

# ---------------------------------------------------------------------------
# Model definition — MODIFY THIS
# ---------------------------------------------------------------------------

MODEL = LogisticRegression(random_state=42, max_iter=1000)

# ---------------------------------------------------------------------------
# Training loop (structure is stable, but agent can modify)
# ---------------------------------------------------------------------------

def main():
    start = time.time()

    # Load data
    df = pd.read_csv(DATA_DIR / "trials.csv")

    # Build features
    X, y = build_features(df)
    feature_names = list(X.columns)

    # Train/val split
    val_ids_path = DATA_DIR / "val_ids.json"
    if val_ids_path.exists():
        with open(val_ids_path) as f:
            val_ids = set(json.load(f))
        train_mask = ~df["nct_id"].isin(val_ids)
    else:
        train_mask = pd.Series([True] * len(df))

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train
    MODEL.fit(X_scaled, y_train)

    # CV score (informational only — real eval is in prepare.py)
    cv = cross_val_score(MODEL, X_scaled, y_train, cv=5, scoring="roc_auc")

    elapsed = time.time() - start

    # Save model bundle
    bundle = {
        "model": MODEL,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # Output in grep-friendly format
    print(f"cv_auc_roc:        {np.mean(cv):.6f}")
    print(f"cv_auc_std:        {np.std(cv):.6f}")
    print(f"train_n_features:  {len(feature_names)}")
    print(f"train_n_samples:   {len(X_train)}")
    print(f"training_seconds:  {elapsed:.1f}")
    print(f"train_model_type:  {type(MODEL).__name__}")

    print(f"\nSelected features:")
    for f in feature_names:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
