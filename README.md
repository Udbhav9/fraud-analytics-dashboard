# 🛡️ E-Commerce Forensic Transaction Analytics & Fraud Detection System

An end-to-end data analytics and machine learning pipeline designed to uncover behavioral fraud signatures across **590,000+ real-world e-commerce transactions** (IEEE-CIS Fraud Detection dataset) and deploy a real-time risk scoring engine.

Unlike standard black-box classification projects, this system prioritizes **investigative forensic Exploratory Data Analysis (EDA)** first—translating domain-level discoveries directly into engineered behavioral features, time-validated gradient boosting models, and SHAP explainability.

---

## 📋 Table of Contents
- [Project Overview & Architecture](#-project-overview--architecture)
- [Key Analytical Discoveries (Forensic EDA)](#-key-analytical-discoveries-forensic-eda)
- [Hypothesis-Driven Feature Engineering](#-hypothesis-driven-feature-engineering)
- [Detection Model & Imbalanced Evaluation](#-detection-model--imbalanced-evaluation)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)

---

## 🏗️ Project Overview & Architecture

Financial transaction datasets present severe class imbalance (~3.5% fraud rate) and complex temporal dependencies. This pipeline solves these challenges across five modular stages:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                     IEEE-CIS RAW DATA MATRIX (590K+ Rows)                  │
│             (train_transaction.csv + train_identity.csv)                   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1 & 2: INGESTION & PROFILING                     │
│  • Dynamic Integer/Float Downcasting (60%+ RAM Footprint Reduction)        │
│  • Missingness Anatomy & Identity Join Correlation Audit                   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: DEEP FORENSIC EDA MODULES                      │
│  • Temporal & Sleep-Cycle Exploitation (Hourly Fraud Likelihood Ratios)    │
│  • Purchaser vs. Recipient Email & Billing Address Discrepancies           │
│  • Hardware, OS & Mobile vs. Desktop Channel Vulnerability Profiling       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  STAGE 4: HYPOTHESIS-DRIVEN ENGINEERING                    │
│  • Synthetic Card UID Reconstruction                                       │
│  • Rolling Transaction Velocity (1-Hour & 24-Hour Windows)                 │
│  • Historical User Spend Deviation Z-Scores & Frequency Encoding           │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               STAGE 5: DETECTION ENGINE & STREAMLIT DEPLOYMENT             │
│  • Chronological Time-Split XGBoost Classifier (PR-AUC Optimized)          │
│  • TreeSHAP Attribution & Cost-Optimal Decision Thresholding               │
│  • Live Interactive Web Dashboard (Forensic Charts + Real-Time Scoring)    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Key Analytical Discoveries (Forensic EDA)

Before training any model, extensive multi-dimensional analysis revealed distinct behavioral signatures separating fraudulent attacks from legitimate buyer activity:

1. **Sleep-Cycle Exploitation (Temporal Spikes):**
   * Fraud likelihood peaks sharply between **2:00 AM and 5:00 AM**, exploiting off-peak hours when cardholders are asleep and automated SMS/email fraud alerts go unnoticed.
2. **Missingness as an Intentional Signal:**
   * Only ~24% of transactions contain accompanying device identity logs (`train_identity`). Transactions exhibiting specific patterns of missing device or recipient metadata display significantly elevated fraud ratios compared to complete logs.
3. **Entity & Domain Discrepancies:**
   * Transactions where the purchaser email domain (`P_emaildomain`) differs from the recipient domain (`R_emaildomain`), or originates from high-risk webmail providers, carry substantially higher chargeback rates.
4. **Channel Vulnerability:**
   * Mobile web channels exhibit systematically higher fraud probabilities than desktop sessions due to device spoofing and emulated user agents.

---

## ⚙️ Hypothesis-Driven Feature Engineering

Every insight uncovered during EDA was translated into concrete predictive features:

* **Synthetic Entity Reconstruction (`Card_UID`):** Combined invariant card issuer, bank, billing zip code, and purchaser email attributes to track unique user cards across time.
* **Rolling Velocity Metrics:** Calculated transaction counts per `Card_UID` over **1-hour** and **24-hour** windows to capture automated card-testing micro-transactions and rapid spending sprees.
* **Spend Deviation ($Z$-Score & Mean Ratios):** Measured how drastically a transaction amount deviates from a specific card's historical spending baseline:
  $$Z = \frac{\text{TransactionAmt} - \mu_{\text{Card\_UID}}}{\sigma_{\text{Card\_UID}} + \epsilon}$$
* **Discrepancy Flags & Frequency Encoding:** Created binary mismatch indicators (`email_domain_mismatch`) and frequency maps for high-cardinality fields (`card1`, `addr1`, `DeviceInfo`).

---

## 📈 Detection Model & Imbalanced Evaluation

To prevent temporal data leakage, the dataset was split **chronologically (80% past training / 20% future validation)** rather than using random shuffling.

### Performance Metrics (Out-of-Time Validation Set)

| Metric | Score | Significance |
| :--- | :---: | :--- |
| **PR-AUC** *(Primary Metric)* | **0.5096** | **~15x improvement** over baseline random guessing (0.0344) |
| **ROC-AUC** | **0.8873** | Strong risk-ranking separation across future transactions |
| **Optimal Threshold** | **0.7730** | Tuned via Precision-Recall F1 maximization |
| **Precision (Fraud Class)** | **56.81%** | Over half of all auto-flagged transactions are true positives |
| **Recall (Fraud Class)** | **48.23%** | Captures nearly half of all fraud with <1.4% false positive rate |

### Model Interpretability (SHAP)
Using **TreeSHAP**, model predictions were audited back to individual input features, confirming that engineered velocity counts, amount deviation ratios, and temporal signals serve as primary decision drivers.

---

## 💻 Interactive Streamlit Dashboard

The project includes a two-page interactive web application (`app.py`):

1. **Page 1 — Forensic EDA & Analytics:** Interactive Plotly visualizations showcasing hourly attack distributions, email domain risk rankings, financial exposure KPIs, and hardware channel breakdowns.
2. **Page 2 — Real-Time Risk Scoring Engine:** Allows users to simulate live transaction payloads (amount, velocity, email domains, device type, transaction hour) and receive real-time probability gauges alongside actionable operational recommendations (*Auto-Approve*, *Step-Up 3DS Auth*, or *Hard Block*).

---

## 📂 Repository Structure

```text
E-Commerce-Forensic-Fraud-Analytics/
│
├── .gitignore                      # Excludes large raw CSV datasets
├── README.md                       # Project documentation & analytical summary
├── requirements.txt                # Python dependencies
├── app.py                          # Streamlit web dashboard application
│
├── notebooks/
│   └── forensic_fraud_pipeline.ipynb  # Complete 5-step Colab analysis & modeling notebook
│
└── dashboard_assets/               # Lightweight model & sampled data artifacts
    ├── xgb_fraud_model.pkl.gz.part1
    ├── xgb_fraud_model.pkl.gz.part2
    ├── xgb_fraud_model.pkl.gz.part3
    ├── model_features.pkl
    ├── cat_cols.pkl
    └── sample_eda.parquet
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/E-Commerce-Forensic-Fraud-Analytics.git
cd E-Commerce-Forensic-Fraud-Analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard Locally
```bash
streamlit run app.py
```

### 4. (Optional) Re-run Full Training Pipeline on Raw Data
Raw CSV files (`train_transaction.csv` and `train_identity.csv`) are excluded from GitHub due to file size limits. To reproduce the full training pipeline from scratch:
1. Download the dataset via the [Kaggle IEEE-CIS Fraud Detection Competition](https://www.kaggle.com/c/ieee-fraud-detection).
2. Place the CSV files in a `data/` directory or run the provided notebook in Google Colab using your Kaggle API token.

---

## 🛠️ Tech Stack
* **Data Manipulation & Memory Optimization:** Python, Pandas, NumPy, PyArrow
* **Machine Learning & Explainability:** XGBoost, Scikit-Learn, SHAP
* **Visualization & Deployment:** Streamlit, Plotly, Matplotlib, Seaborn
