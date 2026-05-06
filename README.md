<div align="center">

<h1>📈 IPO Listing Gain Predictor</h1>

<p><em>A regularised deep neural network that predicts whether an Indian IPO will list at a premium on debut day — built with PyTorch.</em></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

| Metric | Score |
|--------|-------|
| 🎯 Test Accuracy | **66.2%** |
| 🔬 Precision | **70.2%** |
| 📡 Recall | **80.3%** |
| ⚖️ F1 Score | **74.8%** |

</div>

---

## 🗂️ Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Web App](#-web-app)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

The Indian IPO market sees hundreds of listings annually. Retail investors often subscribe to IPOs hoping to capture the **"listing gain"** — the difference between the issue price and the first-day market price. Without a structured, data-driven framework, decisions rely on hearsay, broker tips, and informal Grey Market Premium (GMP) signals.

This project builds an **end-to-end binary classification pipeline** that predicts whether an Indian IPO will list at a **positive premium** using 9 publicly available features extracted from subscription data and issue details — all available *before* listing day.

> **Not financial advice.** This is an academic ML project. Never make investment decisions based solely on model output.

---

## ❓ Problem Statement

**Input:** 9 features available after the IPO subscription window closes (T-2 before listing)  
**Output:** Binary label — `1` (listing gain > 0%) or `0` (listing at flat/discount)  
**Goal:** Build a model with genuine discriminative power beyond a naive majority-class heuristic

### Why is this hard?
- GMP data — the strongest known predictor (r ≈ 0.45–0.60) — is traded informally and not programmatically available
- Subscription multiples can be inflated by HNI leverage and QIB mandated participation
- Market regimes shift: the historical dataset shows ~65% profitable listings, but early 2026 data shows only ~29%

---

## ✨ Features

- **4-layer feedforward DNN** with BatchNorm, Dropout, and early stopping — trained end-to-end in PyTorch
- **4-model benchmark suite**: Dummy baseline → Logistic Regression → Random Forest → Neural Network
- **Stratified 60/20/20 split** preserving class balance across train, validation, and test sets
- **Temporal (chronological) split validation** to detect data leakage from shared market eras
- **Random Forest feature importance** (MDI) to validate and explain feature contributions
- **Browser-based inference UI** (IPO) for live predictions before listing day

---

## 🧠 Model Architecture

```
Input (9 features)
        │
  ┌─────▼──────┐
  │  Linear    │  9 → 64
  │  BatchNorm │
  │  ReLU      │
  │  Dropout   │  p = 0.25
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Linear    │  64 → 32
  │  ReLU      │
  │  Dropout   │  p = 0.125
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Linear    │  32 → 16
  │  ReLU      │
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Linear    │  16 → 1
  │  Sigmoid   │  → P(listing gain) ∈ [0, 1]
  └────────────┘
```

| Component | Choice | Reason |
|-----------|--------|--------|
| Loss | `BCELoss` | Binary classification with sigmoid output |
| Optimiser | `Adam (lr=0.0075)` | Adaptive per-parameter LR; robust to sparse gradients |
| Batch size | `16` | Noisy gradient estimates act as implicit regularisation |
| Regularisation | `BatchNorm + Dropout` | Reduces train-val gap from ~12pp → <5pp |
| Early Stopping | `patience=20` | Restores best checkpoint; stopped at epoch ~47 |
| Total parameters | `~3,000` | Intentionally small — dataset size (~333 IPOs) limits capacity |

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | `ipo_model_dataset_historical_ready.csv` |
| Records | ~333 historical Indian IPOs (zero missing values) |
| Target balance | ~65% profitable / ~35% not profitable |
| Split | 60% train · 20% validation · 20% test (stratified) |
| Supplementary | `ipo_latest_updates_separate.csv` — 14 IPOs from early 2026 (context only) |

### Feature Set

| # | Feature | Type | MDI Importance |
|---|---------|------|---------------|
| 1 | `Subscription_QIB` | Numeric | **0.198** ← strongest |
| 2 | `Issue_Size` | Numeric | 0.162 |
| 3 | `Subscription_RII` | Numeric | 0.151 |
| 4 | `Subscription_HNI` | Numeric | 0.143 |
| 5 | `Issue_Price` | Numeric | 0.138 |
| 6 | `Year` | Temporal | 0.072 |
| 7 | `Month` | Temporal | 0.058 |
| 8 | `Quarter` | Temporal | 0.051 |
| 9 | `DayOfWeek_Enc` | Ordinal | 0.027 |

> **Note:** All features are publicly available from NSE/BSE/SEBI portals after the subscription window closes — no proprietary data required.

---

## 📁 Project Structure

```
ipo-listing-gain-predictor/
│
├── IPO_Return_Enhanced_3.ipynb      # Main notebook — full pipeline
│
├── data/
│   ├── ipo_model_dataset_historical_ready.csv
│   └── ipo_latest_updates_separate.csv
│
├── app/
│   └── ipo/                  # Browser-based inference UI
│
├── outputs/
│   ├── charts/                      # All EDA and results visualisations
│   └── IPO_Listing_Gains_Case_Study_Report.docx
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/RAGHULK08/ipo-listing-gain-predictor.git
cd ipo-listing-gain-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Run the full pipeline (Jupyter)

```bash
jupyter notebook IPO_Return_Enhanced_3.ipynb
```

### Quick inference snippet

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load trained model and scaler
model = torch.load('outputs/reg_model.pt')
scaler = pickle.load(open('outputs/scaler.pkl', 'rb'))
model.eval()

# Input: [QIB, HNI, RII, IssuePrice, IssueSize, Year, Month, Quarter, DayOfWeek_Enc]
sample = np.array([[45.2, 38.7, 22.1, 420, 850, 2025, 12, 4, 2]])

x = torch.tensor(scaler.transform(sample), dtype=torch.float32)

with torch.no_grad():
    prob = model(x).item()

verdict   = "Profitable" if prob > 0.5 else "Not Profitable"
confidence = "Strong Buy" if prob > 0.75 else ("Borderline" if prob > 0.5 else "Avoid")

print(f"Prediction  : {verdict}")
print(f"Probability : {prob*100:.1f}%")
print(f"Confidence  : {confidence}")
# → Prediction  : Profitable
# → Probability : 67.1%
# → Confidence  : Borderline
```

---

## 📈 Results

### Model Comparison (Held-Out Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Dummy Baseline (always profitable) | 65.4% | 65.4% | 100.0% | 79.1% |
| Logistic Regression | 61.8% | 68.1% | 74.8% | 71.3% |
| Random Forest | 63.5% | 69.4% | 78.1% | 73.4% |
| **Neural Network (Regularised) ★** | **66.2%** | **70.2%** | **80.3%** | **74.8%** |

> ★ The Neural Network achieves the highest Accuracy and Precision, and the best non-dummy F1 Score. The Dummy Baseline's inflated F1 (79.1%) is misleading — it predicts *every* IPO as profitable and offers zero filtering value.

### Confusion Matrix (Neural Network)

```
                  Predicted Not Profitable   Predicted Profitable
Actual Not Profitable        28 (TN) ✅              9 (FP) ❌
Actual Profitable            12 (FN) ⚠️             51 (TP) ✅
```

- **FP=9 < FN=12** → conservative bias — the model errs toward capital preservation
- **Precision = 51/60 = 85.0%** on profitable predictions
- **Recall = 51/63 = 80.9%** — finds 81 of every 100 profitable IPOs

### Regularisation Effect

| Metric | Baseline (no reg) | Regularised |
|--------|-------------------|-------------|
| Train-Val accuracy gap | ~12 pp | <5 pp |
| Early stopping epoch | — | ~47 |
| Best Val Accuracy | ~67% | **68.4%** |
| Overfitting | Significant | Controlled |

---

## 🌐 Web App

**IPO** — a browser-based inference interface that accepts the 9 model features and returns a live prediction before listing day.

### Confidence Tiers

| Tier | Probability | Verdict | Suggested Action |
|------|-------------|---------|-----------------|
| 🟢 Strong Buy | P > 75% | Profitable | Subscribe at full allotment |
| 🟡 Borderline | 50% – 75% | Profitable | Subscribe cautiously / await GMP |
| 🔴 Avoid | P < 50% | Not Profitable | Skip subscription |

**Real-time workflow:**

```
IPO Opens (T-5)  →  Subscription Closes (T-2)  →  Model Inference (T-1)  →  Listing Day (T)
                         Final QIB/HNI/RII              P(gain) score
                         multiples published              displayed in UI
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **No GMP data** — strongest predictor (r≈0.45–0.60) unavailable | Primary accuracy ceiling |
| 2 | **Regime shift** — 2026 premium rate dropped from ~65% to ~29% | Model may overestimate gains |
| 3 | **No macro features** — NIFTY50 momentum, repo rate, FII flows excluded | Misses macro-driven signals |
| 4 | **Small dataset** (~333 IPOs) | Limits DNN advantage over tree models |
| 5 | **Single temporal split** using LR proxy | Walk-forward CV would be more rigorous |

### Roadmap

- [ ] **High** — Integrate real GMP via web scraping (`ipowatch.in`, `chittorgarh.com`) → est. +8–12pp accuracy
- [ ] **High** — Quarterly retraining pipeline for regime adaptability
- [ ] **Medium** — Add macro features: NIFTY50 30-day momentum, repo rate delta, FII net flows
- [ ] **Medium** — SHAP explanations for per-IPO feature attribution
- [ ] **Medium** — Walk-forward temporal cross-validation
- [ ] **Low** — Regression variant: predict exact listing gain percentage (for position sizing)
- [ ] **Low** — FastAPI REST inference endpoint
- [ ] **Low** — LSTM / Transformer over chronological IPO sequences

---

## 🧑‍💻 Author

**Raghul K**  
Integrated M.Tech CSE (Data Science)

[![GitHub](https://img.shields.io/badge/GitHub-RAGHULK08-181717?style=flat-square&logo=github)](https://github.com/RAGHULK08)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-raghul--k08-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/raghul-k08)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
