#!/usr/bin/env python3
"""
Predict clinical trial outcome.

Usage:
  python predict.py NCT02578680                    # by NCT ID (must be in dataset)
  python predict.py --search pembrolizumab         # search dataset by drug name
  python predict.py --all                          # predict all validation trials
"""
import sys, os, json, pickle
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle['feature_names']


def predict_trial(df, idx, model, scaler, feature_names):
    """Predict a single trial from the dataset."""
    from train import build_features

    row = df.iloc[idx]
    X, _ = build_features(df)
    X_row = X.iloc[[idx]].copy()

    for f in feature_names:
        if f not in X_row.columns:
            X_row[f] = 0
    X_row = X_row[feature_names]

    X_scaled = scaler.transform(X_row)
    prob = model.predict_proba(X_scaled)[0][1]

    drug = row.get('intervention_name', '?')
    condition = row.get('condition', '?')
    phase = row.get('phase', '?')
    indication = row.get('indication_area', '?')
    label = row.get('label', '?')

    bar_len = int(prob * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    if prob >= 0.7:
        verdict = "LIKELY SUCCESS"
    elif prob >= 0.4:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY FAILURE"

    print(f"  {row.get('nct_id', '?')} | {drug[:30]:30s} | {condition[:25]:25s}")
    print(f"  Phase {phase} | {indication} | actual={label}")
    print(f"  [{bar}] {prob:.1%} → {verdict}")
    print()
    return prob


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'trials.csv'))
    model, scaler, feature_names = load_model()

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python predict.py NCT02578680")
        print("  python predict.py --search pembrolizumab")
        print("  python predict.py --all")
        sys.exit(1)

    if args[0] == '--all':
        with open(os.path.join(DATA_DIR, 'val_ids.json')) as f:
            val_ids = json.load(f)
        val_mask = df['nct_id'].isin(val_ids)
        print(f"Predicting {val_mask.sum()} validation trials:\n")
        for idx in df[val_mask].index:
            predict_trial(df, idx, model, scaler, feature_names)

    elif args[0] == '--search':
        query = ' '.join(args[1:]).lower()
        matches = df[
            df['intervention_name'].str.lower().str.contains(query, na=False) |
            df['condition'].str.lower().str.contains(query, na=False) |
            df['nct_id'].str.contains(query.upper(), na=False)
        ]
        if matches.empty:
            print(f"No trials matching '{query}' in dataset.")
        else:
            print(f"Found {len(matches)} matching trials:\n")
            for idx in matches.index[:10]:
                predict_trial(df, idx, model, scaler, feature_names)

    else:
        for nct_id in args:
            match = df[df['nct_id'] == nct_id]
            if match.empty:
                print(f"  {nct_id} not found in dataset.")
            else:
                predict_trial(df, match.index[0], model, scaler, feature_names)


if __name__ == '__main__':
    main()
