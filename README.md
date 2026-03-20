# ct-predict

Clinical trial outcome predictor. Given a dataset of Phase 2/3 clinical trials with biomedical features, an autonomous ML agent optimizes a model to predict success or failure.

Uses the [autoresearch](https://github.com/karpathy/autoresearch) pattern — an AI agent modifies `train.py`, trains, evaluates, keeps improvements, discards regressions, and repeats.

## Quick start

```bash
# Train the baseline model
python train.py

# Evaluate on held-out set
python prepare.py

# Run autoresearch (autonomous optimization)
claude -p "$(cat program.md)"
```

## How it works

`train.py` is the only mutable file. The autoresearch agent modifies it — trying different feature engineering, models, and hyperparameters. `prepare.py` evaluates on a frozen held-out set. Git tracks every experiment: improvements are committed, regressions are reverted.

## Files

```
ct-predict/
├── program.md     # Agent instructions (the research program)
├── train.py       # Feature engineering + model (agent modifies this)
├── prepare.py     # Evaluation (frozen — DO NOT MODIFY)
├── predict.py     # Predict trials in the dataset
└── data/
    ├── trials.csv     # 2,151 trials × 112 features
    └── val_ids.json   # Held-out validation split
```

## Dataset

2,151 real Phase 2/3 clinical trials across 8 therapeutic areas (oncology, CNS, immunology, metabolic, cardiovascular, respiratory, infectious, other). Features extracted from 35 biomedical data sources covering drug pharmacology, target genetics, pathway biology, protein structure, disease ontology, publication signals, regulatory status, and healthcare economics.
