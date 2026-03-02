# Product Length Predictor
### NLP Regression — Amazon ML Challenge 2023 Dataset

---

## Overview

This project tackles the problem of predicting the physical length of a product given its text description. The dataset originates from the Amazon ML Challenge 2023 and contains approximately 2.2 million product listings. The task is a regression problem where the input is unstructured product text and the output is a continuous numeric value representing product length.

The approach is inspired by the winning solution of the Amazon ML Challenge 2023 and uses semantic similarity search combined with a LightGBM meta model — no fine-tuning required.

---

## Dataset

**Source:** Amazon ML Challenge 2023 (2.2 million rows)

**Columns used:**
- `TOTAL_SENTENCE` — concatenation of product title, bullet points, and description
- `PRODUCT_ID_TYPE` — product category identifier
- `PRODUCT_LENGTH` — target variable (physical length of product)

**Sampling strategy:**
- Full dataset split 90:5:5 into train, val, test using Polars for memory efficiency
- 60,000 rows sampled from the train for local EDA and experimentation
- Sampling guaranteed inclusion of top-50 and bottom-50 rows by `PRODUCT_LENGTH` to preserve extreme values

---

## Exploratory Data Analysis

### Target Distribution
`PRODUCT_LENGTH` showed a right-skewed distribution with skewness of 2.7 and kurtosis of 9.8, indicating the presence of extreme outliers at the upper end. A log1p transformation was applied to the target during training to handle this skew.

### Text Quality
- 15% of descriptions contained HTML tags or entities — cleaned during preprocessing
- No null or empty descriptions found
- 21 duplicate descriptions with differing `PRODUCT_LENGTH` values identified — label noise rate of 0.04%, negligible

### Language Distribution
- ~55,000 rows detected as English
- ~2,000 French, ~1,000 German, ~670 Spanish
- ~7,000 rows had low confidence detection (< 0.5) — likely code-mixed text
- Mean language detection confidence: 0.77

### Text vs Target Correlation
Pearson correlation between word count / character count and `PRODUCT_LENGTH` was near zero. This was the most important EDA finding — it confirmed that surface-level text features carry no predictive signal for length. Semantic understanding of the description is required, ruling out TF-IDF and classical tabular approaches.

### Unit Keyword Analysis
- 22% of descriptions contained dimensional unit keywords (cm, inch, mm, feet, meter)
- Correlation between extracted numeric unit values and `PRODUCT_LENGTH` was effectively zero (0.02 for cm, -0.0009 for inch)
- Sellers use inconsistent unit systems and describe multiple dimensions in the same text — regex extraction is unreliable as a direct predictor

### Class Imbalance
No class imbalance issue in `PRODUCT_ID_TYPE`. The largest single category covered only 5% of data. Top 5 categories together covered 14%. Stratified sampling was not required.

---

## Feature Engineering

### Text Cleaning
- Stripped HTML tags and HTML entities using regex
- Removed emojis (unicode ranges) and special characters
- Normalised literal `\n` and `\r` to spaces
- Lowercased and collapsed multiple whitespace

### Auxiliary Regex Features

| Feature | Description |
|---|---|
| `HAS_UNIT` | Binary — does description mention any unit |
| `UNIT_COUNT` | Number of unit mentions in description |
| `FIRST_NUM` | First numeric value in description |
| `MAX_NUM` | Largest numeric value in description |
| `NUM_COUNT` | Total count of numeric values |
| `FIRST_CM` | First value next to 'cm' keyword |
| `FIRST_INCH` | First value next to 'inch' keyword |
| `HAS_DIM_KEYWORD` | Binary — mentions length/width/height etc. |
| `DIM_KEYWORD_COUNT` | Count of dimension keyword mentions |
| `WORD_COUNT` | Number of words in description |
| `CHAR_COUNT` | Character count of description |
| `UNIT_TYPE_ENC` | Dominant unit system: 0=none, 1=metric, 2=imperial, 3=both |

> Note: Correlation of `FIRST_CM` and `FIRST_INCH` with `PRODUCT_LENGTH` was near zero. These served as weak auxiliary signals only. The main predictive power came from KNN neighbour features.

---

## Modelling Approach

### Architecture Overview

A two-stage system:
1. **Semantic nearest neighbour retrieval** using pretrained sentence transformers
2. **LightGBM meta model** that learns to predict product length from neighbour target values

**Core intuition:** if a product description is semantically similar to another product in the training corpus, they likely have similar physical lengths. Rather than predicting length directly from text, the model finds the most similar products it has seen and learns a complex aggregation over their lengths.

### Data Split for KNN Feature Generation

To avoid mutual neighbour leakage — where two similar points appear in each other's neighbour lists and leak their targets into each other's features:

- **45,000 rows** → corpus (search database only, never used for LightGBM training)
- **15,000 rows** → query split (features generated from corpus, used for LightGBM training)

Val and test sets also searched against the same 45k corpus. No target leakage of any kind.

### Embedding Models

| Model | Size | Why chosen |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 118MB | Fast, battle-tested on product similarity, handles non-English rows |
| `intfloat/multilingual-e5-small` | 118MB | More modern training objective, diverse neighbours from MiniLM |

- Descriptions truncated to 2000 characters (~512 tokens) for speed
- Embeddings L2-normalised before search — cosine similarity reduced to inner product
- **50 nearest neighbours per model = 100 total KNN features**

### Nearest Neighbour Search

- FAISS `IndexFlatIP` (exact inner product search) on GPU
- Exact KNN used — no approximation needed at 60k scale, runs in seconds on T4
- ANN (HNSW or IVF) would only be needed at 1M+ rows

### Meta Model — LightGBM

LightGBM regressor trained on 100 KNN neighbour lengths + 12 auxiliary features + `PRODUCT_ID_TYPE` (categorical).

| Parameter | Value |
|---|---|
| Objective | `mape` |
| n_estimators | 5000 with early stopping (patience=100) |
| learning_rate | 0.05 |
| num_leaves | 127 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Target clipping | min=10 for MAPE numerical stability |

### Training Protocol
- 5-fold cross-validation on combined query + val meta sets (20,000 rows)
- Out-of-fold predictions collected for honest evaluation
- Final model trained on full 20k rows using mean best iteration from CV folds
- Final predictions on held-out test set (10,000 rows)

---

## Results

| Metric | Value |
|---|---|
| Test MAPE | 11.57% |
| Test Competition Score | **88.43** |
| Formula | `max(0, 100 * (1 - MAPE))` |

---

## Technology Stack

| Library | Purpose |
|---|---|
| Polars | Memory-efficient loading of 2.2M row CSV |
| Pandas | Local EDA and feature engineering |
| sentence-transformers | Pretrained multilingual embeddings |
| FAISS (GPU) | Exact nearest neighbour search |
| LightGBM | Meta model regression with MAPE objective |
| FastText LID | Language detection during EDA |
| scikit-learn | Metrics, train/test splitting |
| NumPy / SciPy | Numerical operations, statistical tests |
| Matplotlib / Seaborn | EDA visualisations |

---

## Key Learnings

- Zero correlation between text length and target is a strong signal that semantic modelling is required — surface features will not work
- The KNN-embedding approach reframes regression as similarity retrieval — instead of predicting a number, find similar things and learn from their labels
- Mutual neighbour leakage is a subtle but real data leakage risk in KNN feature generation — corpus/query separation is the correct fix
- MAPE objective in LightGBM requires target clipping at a minimum value — very small targets cause numerical instability
- Frozen pretrained embeddings with a learned meta model can outperform naive approaches while being significantly faster to iterate

---

## Future Work

- Replace MiniLM with `BAAI/bge-m3` (8192 token context, hybrid dense+sparse retrieval) for better handling of long descriptions
- Fine-tune `DeBERTa-v3-base` with a regression head as a stronger baseline
- Hybrid approach: combine KNN neighbour features with DeBERTa embeddings as input to the meta model
- Increase K from 50 to 100 neighbours per model for richer LightGBM features
- Add statistical aggregations of neighbour lengths (mean, median, std, min, max) as explicit features
