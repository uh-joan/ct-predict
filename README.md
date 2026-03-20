# ct-predict

Clinical trial outcome predictor. Predicts whether a Phase 2/3 clinical trial will succeed or fail based on drug, target, and disease features extracted from biomedical databases.

**0.90 AUC-ROC** on 431 held-out trials across 8 therapeutic areas. ~85% accuracy on 40+ hand-tested unseen trials including famous failures (solanezumab, bapineuzumab, ECHO-301) and successes (semaglutide, dupilumab, lecanemab).

## How it works

The dataset (`data/trials.csv`) contains 2,151 real Phase 2/3 clinical trials with features from 35 biomedical data sources — drug pharmacology (ChEMBL, DrugBank), target genetics (gnomAD, GWAS, ClinVar), protein structure (PDB, AlphaFold), pathway biology (Reactome, KEGG), disease ontology (HPO, COSMIC), publication signals (PubMed, OpenAlex), regulatory data (FDA, EMA), and healthcare spending (Medicare, Medicaid).

An autonomous ML agent ([autoresearch](https://github.com/karpathy/autoresearch) pattern) modifies `train.py` to improve the model — trying different algorithms, feature engineering, hyperparameters — keeping only improvements and discarding regressions. 130+ experiments ran autonomously to reach the current model.

## Usage

```bash
# Predict trials in the dataset
python predict.py --search pembrolizumab
python predict.py --search "semaglutide obesity"
python predict.py NCT02752074

# Predict all validation trials
python predict.py --all

# Train the model
python train.py

# Evaluate on held-out set
python prepare.py
```

## Autoresearch

The agent reads `program.md`, modifies `train.py`, trains, evaluates, and iterates:

```bash
claude -p "$(cat program.md)"
```

The loop:
1. Modify `train.py` (feature engineering, model, hyperparameters)
2. `python train.py > run.log 2>&1`
3. `python prepare.py >> run.log 2>&1`
4. `grep "^auc_roc:" run.log`
5. If improved → `git commit`, if worse → `git reset --hard HEAD~1`
6. Repeat

`prepare.py` is frozen — the agent cannot modify the evaluation. `train.py` is the only mutable file.

## Files

```
ct-predict/
├── program.md     # Autoresearch agent instructions
├── train.py       # Feature engineering + model (agent modifies this)
├── prepare.py     # Frozen evaluation (DO NOT MODIFY)
├── predict.py     # Inference on dataset trials
└── data/
    ├── trials.csv     # 2,151 trials × 112 features
    └── val_ids.json   # Held-out validation split
```

## Dataset

2,151 clinical trials across 8 therapeutic areas:

| Area | Trials | Success Rate |
|------|--------|-------------|
| Oncology | 808 | 55% |
| CNS | 267 | 57% |
| Immunology | 232 | 88% |
| Other | 231 | 63% |
| Metabolic | 178 | 69% |
| Respiratory | 155 | 66% |
| Cardiovascular | 141 | 66% |
| Infectious | 139 | 62% |

Features include trial design (phase, enrollment, endpoint type, num sites), drug properties (ChEMBL selectivity, max phase, mechanism count), target biology (gnomAD pLI, Reactome pathways, COSMIC mutations, PDB structures), disease evidence (OpenTargets score, GWAS hits, HPO phenotypes), publication signals (PubMed, OpenAlex, bioRxiv), regulatory status (FDA approval, EMA), healthcare economics (Medicare spend), and combination drug features (drug2 FDA status, target interaction, GO overlap).

## Key features discovered by autoresearch

- **same_target_success_rate**: Historical success rate of drugs targeting the same gene. BACE inhibitors = 0% (all failed) → model correctly flags atabecestat.
- **target_disease_score**: OpenTargets evidence for target + specific disease. JAK1+RA = 0.67 (strong), JAK1+lupus = 0 (none) → model sees indication-specific risk.
- **combo_drug2_target_disease_score**: For combinations, evidence strength of the added drug's target for this disease.
- **target_expression_breadth**: GTEx tissues with TPM > 1. Ubiquitous targets (54/54) have higher off-target safety risk than specific ones (16/54).

## Results on unseen trials

| Trial | Prediction | Actual |
|-------|-----------|--------|
| Solanezumab (Alzheimer) | 20-31% | Failed ✓ |
| Bapineuzumab (Alzheimer) | 18% | Failed ✓ |
| Atabecestat (Alzheimer) | 21-27% | Failed ✓ |
| Lecanemab (Alzheimer) | 72-76% | Succeeded ✓ |
| Lorcaserin (obesity) | 29-37% | Withdrawn ✓ |
| Semaglutide (obesity) | 87-91% | Succeeded ✓ |
| Dupilumab (atopic dermatitis) | 79% | Succeeded ✓ |
| Tofacitinib (RA) | 90% | Succeeded ✓ |
| Bedaquiline (TB) | 88% | Succeeded ✓ |
| IMvigor211 (atezo bladder) | 20% | Failed ✓ |
| Risdiplam (SMA) | 78% | Succeeded ✓ |
