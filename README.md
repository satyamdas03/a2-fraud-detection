# Credit Card Fraud Detection - UTS 32513 A2

Public Colab notebook: https://colab.research.google.com/github/satyamdas03/a2-fraud-detection/blob/main/notebook/fraud_detection.ipynb

## What This Project Does

Builds a cost-sensitive LightGBM classifier to detect fraudulent credit card transactions. Uses the ULB Credit Card Fraud Detection dataset (284,807 transactions, 0.172% fraud rate).

## Key Design Points

- Temporal train/validation/test split to avoid look-ahead leakage.
- LightGBM with class balancing and randomized hyperparameter search.
- Threshold chosen on validation data to minimize a custom asymmetric cost function.
- Evaluation uses AUROC, average precision, F1, precision@100, and total cost — not accuracy.

## Local Reproduction

```bash
pip install numpy pandas scikit-learn lightgbm scipy matplotlib
python run_pipeline.py
```

The dataset is downloaded automatically if it is not already in `data/creditcard.csv`.

## Files

- `notebook/fraud_detection.ipynb` — self-contained Colab notebook.
- `report/combined_report.pdf` — submission journal (regenerate from markdown if needed).
- `presentation/slides.md` — A3 presentation outline.
- `utils/` — modular data loader, model wrapper, and evaluation helpers.
