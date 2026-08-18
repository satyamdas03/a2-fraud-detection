# Implementation Log

## 1. Initial Design Decisions

I started by reading the A2 specification and the subject outline. The key constraints were:

- Option 2 requires a real, non-trivial dataset.
- The notebook must be self-contained and runnable on a cloud platform.
- The assessment is 50% code/report and 20% presentation/peer review; attendance at A3 is mandatory.
- AI tools are allowed but must be documented and critically reviewed.

I chose credit card fraud detection because it has a clear business objective, a well-known public dataset, and an obvious mismatch between training loss (cross-entropy) and business objective (asymmetric fraud cost). That mismatch gives something concrete to analyze for Criterion C.

## 2. Use of AI Tools

I used Claude Code as the main AI assistant for this project. The interactions fell into four categories:

### 2.1 Specification interpretation and project planning

I pasted the A2 PDF and subject outline into Claude and asked for a recommendation between Option 1 and Option 2. Claude recommended Option 2 with a real-world dataset and a custom evaluation metric. I accepted this recommendation because it aligned with the rubric: Option 2 allows library use while still requiring deep justification of data/model alignment, and a custom cost metric directly targets the "Evaluation and Refinement" criterion.

### 2.2 Code generation

I asked Claude to generate:

- A `data_loader.py` module for temporal splitting and scaling.
- A `model.py` wrapper around LightGBM with randomized hyperparameter search.
- An `evaluation.py` module with cost-sensitive threshold optimization and plotting.
- A self-contained Colab notebook.

I then ran the full pipeline locally and checked that the shapes, class counts, and metrics were reasonable. For example, I verified that the temporal split produced training/validation/test fraud counts close to 360/57/75, matching the overall 0.172% fraud rate.

### 2.3 Report drafting

I asked Claude to draft the report sections using a direct, human-like tone. I then edited the drafts to remove generic phrasing, inserted the actual numeric results from the pipeline, and added specific design justifications (e.g., why a temporal split matters for fraud detection).

### 2.4 Critical review and verification

Every AI-generated claim was checked against the code or external sources:

- The LightGBM binary cross-entropy gradient formulas were cross-checked with the LightGBM documentation.
- The dataset source and size (284,807 rows, 492 frauds) were verified by loading the CSV directly.
- The claim that tree ensembles outperform neural networks on tabular data was supported by Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on typical tabular data?"

## 3. Challenges and Solutions

### 3.1 Downloading the dataset inside the notebook

The original ULB dataset is hosted on Kaggle, which requires authentication. A public notebook cannot ask the user for Kaggle credentials. I found a direct public URL to the same CSV (`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`) and used `urllib.request.urlretrieve` in the notebook. I verified locally that the file size and row count matched the known dataset.

### 3.2 Hyperparameter search time

A full grid search over the LightGBM space would take too long in a 15-minute presentation demo. I used `RandomizedSearchCV` with only 15 candidates and 3-fold CV. This keeps runtime under a few minutes while still exploring the main hyperparameters.

### 3.3 Class imbalance and metric choice

Accuracy on this dataset is always above 99%, so it is not useful. I added AUROC, average precision, F1, and a custom cost metric. The custom cost weights false negatives 10x more than false positives, which forces the model to prioritize catching fraud over avoiding false alarms.

### 3.4 Threshold optimization

The default 0.5 threshold is inappropriate for imbalanced data. I wrote a function that scans thresholds on the validation set and picks the one minimizing total cost. This is a post-training step, which separates probability calibration from decision making.

## 4. Knowledge Gaps and Verification

| Topic | Confidence | How verified |
|-------|------------|--------------|
| LightGBM gradient formulas | High | Official docs + reproduced in report |
| Temporal split vs. random split | High | Domain knowledge; no code bug because fraud rate stays consistent across splits |
| Cost-weighted threshold selection | High | Implemented from scratch; checked monotonicity of cost function |
| PCA feature interpretation | Low | Features are anonymized; I explicitly state this limitation |
| Exact real-world cost ratios | Medium | Used illustrative ratio (10:1); noted this should be set by business |

## 5. Reflection

The most valuable part of the project was designing the evaluation around the business objective rather than just accuracy. If I had stopped at accuracy, the model would look perfect while missing most frauds. The biggest risk for the presentation is explaining the threshold choice clearly, because peers may expect a default 0.5 cutoff. I will spend time in the slides explaining why 0.004 is the right operating point given the cost structure.

## 6. Tools and Libraries Used

- Python 3.10
- NumPy, pandas, scikit-learn, LightGBM, SciPy, Matplotlib
- Claude Code for code generation, planning, and writing assistance
- Google Colab target platform for the public notebook
