<!-- BANNER -->
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<br/>

<h1 align="center">🛒 Online Retail Customer Segmentation</h1>
<h3 align="center">RFM Feature Engineering + KMeans Clustering in Python</h3>

<p align="center">
  <a href="https://www.youtube.com/watch?v=afPJeQuVeuY">
    <img src="https://img.youtube.com/vi/afPJeQuVeuY/maxresdefault.jpg" alt="Watch the Tutorial" width="600"/>
  </a>
  <br/>
  <em>▶ Click the image above to watch the full tutorial on YouTube</em>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
  - [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [Data Cleaning](#2-data-cleaning)
  - [KMeans Clustering Theory](#3-how-kmeans-clustering-works)
  - [Feature Engineering (RFM)](#4-rfm-feature-engineering)
  - [KMeans Modeling](#5-kmeans-clustering)
  - [Cluster Analysis & Insights](#6-cluster-analysis--insights)
  - [Outlier Analysis](#7-outlier-analysis)
  - [Visualization](#8-visualization)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Project Chapters (Video Timestamps)](#-project-chapters)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🔍 Overview

This project applies **KMeans Clustering** to a real-world UK online retail dataset to uncover meaningful customer segments. By engineering RFM (Recency, Frequency, Monetary) features from raw transactional data, we generate actionable insights about customer behavior — helping businesses improve targeting, retention, and overall customer experience.

> This project is based on a real client engagement and was originally inspired by a published whitepaper on customer mining. It demonstrates the full data science lifecycle from raw data to business insight.

**Key Learning Outcomes:**
- Loading and exploring large Excel-based datasets with `pandas`
- Performing in-depth EDA including regex-based pattern validation
- Executing multi-step data cleaning with transparent data loss tracking
- Engineering domain-specific features (RFM) for unsupervised learning
- Applying and tuning `sklearn` KMeans clustering
- Performing cluster analysis and interpreting results for business impact
- Visualizing high-dimensional cluster results using dimensionality reduction

---

## 📦 Dataset

| Property        | Details                                                                  |
|-----------------|--------------------------------------------------------------------------|
| **Name**        | Online Retail II                                                         |
| **Source**      | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **Coverage**    | UK-based online retail transactions, 2009–2011                           |
| **Sheet Used**  | 2009–2010 (Year 1)                                                       |
| **Records**     | ~520,000 raw transactions                                                |
| **Currency**    | British Pounds Sterling (£)                                              |

### Dataset Columns

| Column         | Type     | Description                                              |
|----------------|----------|----------------------------------------------------------|
| `Invoice`      | String   | 6-digit invoice number; prefix `C` = cancellation       |
| `StockCode`    | String   | 5-digit product code (may include letter suffix)         |
| `Description`  | String   | Product name/description                                 |
| `Quantity`     | Integer  | Units purchased per transaction line                     |
| `InvoiceDate`  | Datetime | Date and time of invoice                                 |
| `Price`        | Float    | Unit price in GBP (£)                                    |
| `Customer ID`  | Float    | Unique customer identifier (nullable)                    |
| `Country`      | String   | Country of customer                                      |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

The EDA phase deeply investigates each column for data quality issues before any cleaning or modeling:

- **Schema inspection** using `df.info()` and `df.describe()` (numeric + object)
- **Null analysis**: `Customer ID` has ~103,000 missing values (~20% of data)
- **Negative quantity detection**: Filtered and identified cancellation invoices (prefix `C`)
- **Invoice pattern validation** using regex `^\d{6}$` — found `C` (cancellations) and `A` (accounting adjustments) prefixes
- **Stock code validation** using regex `^\d{5}$` and `^\d{5}[a-zA-Z]+$` — identified non-product codes (e.g., `DOT`, `M`, `D`, `POST`, `S`)
- **Accounting adjustment records** (prefix `A`): Only 3 records with large negative monetary values — excluded

**Key EDA findings table:**

| Issue Found              | Column       | Action Taken         |
|--------------------------|--------------|----------------------|
| ~103K missing customers  | `Customer ID`| Drop nulls           |
| Negative quantities      | `Quantity`   | Filter via invoice cleaning |
| Negative/zero prices     | `Price`      | Filter `Price > 0`   |
| Cancellation invoices    | `Invoice`    | Regex filter (exclude `C*`) |
| Accounting invoices      | `Invoice`    | Regex filter (exclude `A*`) |
| Non-product stock codes  | `StockCode`  | Regex whitelist only valid codes |

---

### 2. Data Cleaning

A dedicated `cleaned_df` copy is built from the original data with the following sequential steps:

```python
# Step 1: Cast invoice to string for regex filtering
cleaned_df['Invoice'] = cleaned_df['Invoice'].astype(str)

# Step 2: Keep only 6-digit numeric invoices (remove cancellations & accounting)
mask = cleaned_df['Invoice'].str.match(r'^\d{6}$')
cleaned_df = cleaned_df[mask]

# Step 3: Cast StockCode to string
cleaned_df['StockCode'] = cleaned_df['StockCode'].astype(str)

# Step 4: Keep only valid StockCodes (5-digit, 5-digit+letters, or 'PADS')
mask = (
    cleaned_df['StockCode'].str.match(r'^\d{5}$') |
    cleaned_df['StockCode'].str.match(r'^\d{5}[a-zA-Z]+$') |
    cleaned_df['StockCode'].str.match(r'^PADS$')
)
cleaned_df = cleaned_df[mask]

# Step 5: Drop null Customer IDs
cleaned_df.dropna(subset=['Customer ID'], inplace=True)

# Step 6: Remove zero-priced items
cleaned_df = cleaned_df[cleaned_df['Price'] > 0]
```

**Data Retention:**
Original records : ~520,000
Cleaned records : ~400,000
Data retained : ~77%
Data dropped : ~23%


---

### 3. How KMeans Clustering Works

KMeans is an **unsupervised ML algorithm** that partitions data into `K` clusters by iteratively:

1. **Initializing** `K` random centroids in feature space
2. **Assigning** each data point to the nearest centroid (Euclidean distance)
3. **Recomputing** each centroid as the vector mean of its cluster
4. **Repeating** steps 2–3 until centroids converge (minimal movement)

> Analogy: Think of the **Urgency/Importance Matrix** used in time management — tasks are grouped into quadrants based on two features. KMeans does this automatically across N-dimensional feature space.

**Why KMeans for this problem?**
- No labeled customer data available → unsupervised approach required
- Scalable to hundreds of thousands of records
- Interpretable cluster centroids allow actionable business insights
- Well-suited for RFM feature space (low dimensionality, numeric)

---

### 4. RFM Feature Engineering

Three core features are engineered per customer from raw transactions:

| Feature   | Definition                                              | Aggregation           |
|-----------|---------------------------------------------------------|-----------------------|
| **R** – Recency   | Days since last purchase (relative to max date in dataset) | `MAX(InvoiceDate)` → days delta |
| **F** – Frequency | Total number of unique invoices                         | `NUNIQUE(Invoice)`    |
| **M** – Monetary  | Total spend in GBP                                      | `SUM(Quantity × Price)` |

```python
# Line total per transaction
cleaned_df['SalesLineTotal'] = cleaned_df['Quantity'] * cleaned_df['Price']

# RFM aggregation
aggregated_df = cleaned_df.groupby('Customer ID', as_index=False).agg(
    MonetaryValue=('SalesLineTotal', 'sum'),
    Frequency=('Invoice', 'nunique'),
    LastInvoiceDate=('InvoiceDate', 'max')
)

# Recency: days since last purchase (from max date in dataset)
max_invoice_date = aggregated_df['LastInvoiceDate'].max()
aggregated_df['Recency'] = (max_invoice_date - aggregated_df['LastInvoiceDate']).dt.days
```

**Why use RFM?**
- Proven framework for customer value segmentation in retail analytics
- Directly interpretable: high F + high M + low R = high-value active customer
- Reference: [Mining Online Retail Data (Whitepaper)](https://link.springer.com/article/10.1057/dbm.2012.17)

---

### 5. KMeans Clustering

KMeans is applied to the scaled RFM feature matrix using `sklearn`:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Feature matrix
X = aggregated_df[['Recency', 'Frequency', 'MonetaryValue']]

# Standardize features (critical for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K using the Elbow Method
inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=<optimal_k>, random_state=42, n_init=10)
aggregated_df['Cluster'] = kmeans.fit_predict(X_scaled)
```

> **Note:** Feature scaling with `StandardScaler` is mandatory before KMeans — the algorithm is distance-based and highly sensitive to feature magnitude differences.

---

### 6. Cluster Analysis & Insights

After fitting, each cluster is profiled by computing mean RFM values per segment:

```python
cluster_profile = aggregated_df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
```

Typical segments discovered:

| Cluster | Recency  | Frequency | Monetary | Label (Example)       |
|---------|----------|-----------|----------|-----------------------|
| 0       | Low      | High      | High     | 🏆 Champions           |
| 1       | High     | Low       | Low      | 😴 Inactive/Lost       |
| 2       | Medium   | Medium    | Medium   | 🔄 At-Risk / Potential |
| 3       | Low      | Low       | Medium   | 🆕 Recent New Buyers   |

> Exact labels will depend on your K selection and data. Use centroid values to interpret each group.

---

### 7. Outlier Analysis

High-value customers (extreme M or F) can distort cluster boundaries. Outlier analysis is performed to:

- Identify and inspect extreme spenders / high-frequency buyers
- Optionally cap or log-transform `MonetaryValue` and `Frequency` before clustering
- Compare cluster quality metrics (inertia, silhouette score) with and without outliers

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, aggregated_df['Cluster'])
print(f"Silhouette Score: {score:.4f}")
```

---

### 8. Visualization

Cluster results are visualized using multiple approaches:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 2D PCA projection of clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=aggregated_df['Cluster'], palette='tab10', alpha=0.6)
plt.title('Customer Clusters (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
```

Additional visualizations include:
- **Elbow curve** (Inertia vs K) for optimal cluster count selection
- **Box plots** of RFM features per cluster
- **Bar charts** of cluster sizes and average monetary contribution

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- Jupyter Notebook or JupyterLab

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/trentpark8800/online-retail-data-clustering.git
cd online-retail-data-clustering
```

**2. (Recommended) Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
pandas
matplotlib
seaborn
scikit-learn
openpyxl
jupyter


> `openpyxl` is required as the pandas Excel engine to read `.xlsx` files.

**4. Download the dataset:**

Download `online_retail_II.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it inside the `data/` folder.

### Running the Notebook

```bash
jupyter notebook online_retail_data_clustering.ipynb
```

Run all cells sequentially from top to bottom. The notebook is self-contained and follows the project pipeline end-to-end.

---

## 📊 Results

| Metric                    | Value              |
|---------------------------|--------------------|
| Raw records               | ~520,000           |
| Records after cleaning    | ~400,000 (~77%)    |
| Unique customers (RFM)    | ~3,500–4,000       |
| Optimal K (clusters)      | Determined via elbow method |
| Silhouette Score          | Evaluated post-clustering  |
| Currency                  | GBP (£)            |

Key business insights derived from clusters:
- **Champions** — High spend, high frequency, very recent → Reward and retain
- **At-Risk** — Previously active but increasing recency → Re-engagement campaigns
- **Lost/Inactive** — Long recency, low frequency → Win-back or deprioritize
- **New Customers** — Low recency, low frequency → Onboarding nurture

---

## 🛠 Tech Stack

| Library         | Purpose                                |
|-----------------|----------------------------------------|
| `pandas`        | Data loading, cleaning, aggregation    |
| `scikit-learn`  | KMeans clustering, scaling, PCA, metrics |
| `matplotlib`    | Core plotting and visualization        |
| `seaborn`       | Statistical visualizations             |
| `openpyxl`      | Excel file reading engine for pandas   |
| `re` (stdlib)   | Regex pattern validation on invoices/stock codes |

---

## 📺 Project Chapters

| Timestamp    | Chapter                          |
|--------------|----------------------------------|
| `00:00:00`   | Introduction & Project Overview  |
| `00:02:35`   | Environment Setup                |
| `00:05:27`   | Exploratory Data Analysis (EDA)  |
| `00:24:23`   | Data Cleaning                    |
| `00:33:20`   | How KMeans Clustering Works      |
| `00:37:55`   | RFM Feature Engineering          |
| `01:11:59`   | KMeans Clustering                |
| `01:25:40`   | Cluster Analysis                 |
| `01:33:31`   | Outlier Analysis                 |
| `01:41:34`   | Visualization                    |
| `01:46:27`   | Outro & Thanks                   |

---

## 🙏 Acknowledgements

- **Tutorial by:** [TrentDoesMath](https://www.youtube.com/@TrentDoesMath) — Original video walkthrough
- **Dataset:** [UCI ML Repository – Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Whitepaper inspiration:** [Mining Online Retail Customer Data](https://link.springer.com/article/10.1057/dbm.2012.17)
- **Urgency/Importance Matrix image credit:** [Rushcutters Health](https://rushcuttershealth.com.au/how-to-prioritise-tasks-and-get-more-of-the-important-things-done-the-urgent-vs-important-matrix/)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.


---

<div align="center">
  <sub>Built with ❤️ for the data science community · Based on the tutorial by <a href="https://www.youtube.com/@TrentDoesMath">TrentDoesMath</a></sub>
</div>
