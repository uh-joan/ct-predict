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
from sklearn.ensemble import HistGradientBoostingClassifier
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
        # DrugBank
        "drugbank_interaction_count", "drugbank_target_count",
        "drugbank_enzyme_count", "drugbank_transporter_count",
        "drugbank_half_life_hours", "drugbank_molecular_weight",
        # BindingDB
        "bindingdb_ki_nm", "bindingdb_kd_nm", "bindingdb_num_measurements",
        # ClinPGx
        "clinpgx_guideline_count", "clinpgx_actionable", "clinpgx_cyp_substrate_count",
        # FDA
        "fda_prior_approval_class", "fda_breakthrough", "fda_fast_track",
        "fda_orphan", "fda_class_ae_count",
        # Publications
        "pubmed_target_pub_count", "pubmed_drug_pub_count",
        "openalex_citation_velocity", "biorxiv_preprint_count",
        # Healthcare spend
        "medicare_indication_spend", "medicaid_indication_spend",
        # Pathway/network
        "reactome_pathway_count", "stringdb_interaction_degree", "stringdb_betweenness",
        # Genomic
        "gtex_tissue_specificity", "gnomad_pli", "gnomad_loeuf",
        "clinvar_pathogenic_count", "gwas_hit_count", "gwas_best_pvalue",
        "depmap_essentiality", "cbioportal_mutation_freq",
        # Disease
        "hpo_phenotype_count", "monarch_gene_count",
        # Regulatory
        "ema_approved_similar", "eu_filings_count",
        # Structure
        "pdb_structure_count", "pdb_has_ligand_bound",
        "alphafold_available", "alphafold_confidence",
        # Other omics
        "brenda_has_kinetics", "brenda_km_count",
        "cdc_has_surveillance", "nlm_condition_codes",
        "cosmic_is_driver", "cosmic_mutation_count",
        "ensembl_transcript_count", "geo_dataset_count",
        "go_biological_process_count", "go_molecular_function_count", "go_term_count",
        "kegg_pathway_count",
        "ot_disease_association_count", "ot_safety_liability_count",
        "pubchem_complexity", "pubchem_hbond_acceptor", "pubchem_hbond_donor",
        "pubchem_molecular_weight", "pubchem_rotatable_bonds", "pubchem_xlogp",
        # Combo
        "is_combination", "n_drugs", "combo_drug2_has_target",
        "combo_drug2_fda_approved", "combo_drug2_trial_count",
        "combo_drug2_max_phase", "combo_phase_ratio",
        "combo_drug2_pub_count", "combo_drug2_ot_score", "combo_drug2_pli",
        "combo_targets_interact", "combo_shared_pathways", "combo_targets_same_pathway",
        "combo_go_overlap", "combo_bp_overlap", "combo_shared_bp_count",
        "target_expression_breadth",
        "combo_drug2_target_disease_score", "combo_drug2_completed_trials",
        "combo_drug2_terminated_trials", "combo_drug2_fail_ratio",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Missing value handling: indicator + median fill
    for col in available:
        if X[col].isna().mean() > 0.1:
            X[f"{col}_missing"] = X[col].isna().astype(int)
        med = X[col].median()
        X[col] = X[col].fillna(med if pd.notna(med) else 0)

    # One-hot encode categoricals
    if "indication_area" in df.columns:
        for val in df["indication_area"].dropna().unique():
            X[f"is_{val}"] = (df["indication_area"] == val).astype(int)

    if "endpoint_type" in df.columns:
        for val in df["endpoint_type"].dropna().unique():
            X[f"endpoint_{val}"] = (df["endpoint_type"] == val).astype(int)

    return X, y

# ---------------------------------------------------------------------------
# Model definition — MODIFY THIS
# ---------------------------------------------------------------------------

MODEL = HistGradientBoostingClassifier(
    max_iter=500, max_depth=5, learning_rate=0.05,
    min_samples_leaf=16, l2_regularization=1.0, random_state=42,
)

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
