# 🎵 Song Release Year Prediction — End-to-End Regression Pipeline

> **UTS Machine Learning — Individual Task**
> "Hands-On End-to-End Models Machine Learning"

---

## 📌 Identitas

| Field | Detail |
|-------|--------|
| **Nama** | [ISI NAMA KAMU] |
| **NIM** | [ISI NIM KAMU] |
| **Kelas** | Machine Learning |

---

## 🎯 Tujuan Project

Membangun pipeline Machine Learning end-to-end untuk memprediksi **tahun rilis sebuah lagu** berdasarkan fitur-fitur audio (timbre dan karakteristik sinyal musik lainnya).

**Target prediksi:** `year` — tahun rilis lagu (nilai kontinu → **Regression Task**)

---

## 📂 Struktur Repository

```
midterm-machine-learning/
│
├── regresi_song_year_colab.ipynb  ← Notebook utama (semua kode & hasil)
├── model_results.csv              ← Tabel perbandingan semua model
├── target_distribution.png        ← Distribusi tahun rilis
├── feature_distributions.png      ← Distribusi 6 fitur pertama
├── correlation_target.png         ← Korelasi fitur vs target
├── correlation_heatmap.png        ← Heatmap korelasi
├── model_comparison.png           ← Perbandingan metrik semua model
├── actual_vs_predicted.png        ← Scatter actual vs predicted + residual
├── residual_distribution.png      ← Distribusi residual
├── feature_importance.png         ← Feature importance XGBoost
├── optuna_history.png             ← Optuna optimization history
├── lime_explanation_1.png         ← LIME penjelasan prediksi sampel 1
├── lime_explanation_2.png         ← LIME penjelasan prediksi sampel 2
└── README.md                      ← File ini
```

> ⚠️ **Dataset tidak disertakan** karena disediakan oleh dosen.
> Lihat bagian **Dataset** untuk informasi lebih lanjut.

---

## 📊 Dataset

**Nama:** `midterm-regresi-dataset.csv`
**Sumber:** Disediakan oleh dosen melalui link Regresi Dataset

### Deskripsi:
| Kolom | Keterangan |
|-------|-----------|
| Kolom ke-1 | **Target** — tahun rilis lagu (contoh: 2001) |
| Kolom ke-2 dst | **Fitur audio** — nilai numerik dari sinyal musik (timbre, dll), diberi nama `feature_1`, `feature_2`, ... |

---

## 🔧 Alur Pipeline

```
Raw CSV (midterm-regresi-dataset.csv)
       │
       ▼
  Rename kolom → year + feature_1, feature_2, ...
       │
       ▼
  Exploratory Data Analysis (EDA)
  - Distribusi target (year)
  - Missing values analysis
  - Korelasi fitur vs target
  - Heatmap korelasi
       │
       ▼
  Preprocessing & Feature Engineering
  - Outlier handling (IQR clip)
  - Median Imputation
  - Feature interaksi (top-5 fitur)
  - StandardScaler
       │
       ▼
  Train-Test Split (80:20)
       │
       ▼
  Training 7 Baseline Models
  ┌──────────────────────────────────┐
  │ 1. Linear Regression             │
  │ 2. Ridge Regression              │
  │ 3. Lasso Regression              │
  │ 4. Decision Tree Regressor       │
  │ 5. Random Forest Regressor       │
  │ 6. XGBoost Regressor             │
  │ 7. LightGBM Regressor            │
  └──────────────────────────────────┘
       │
       ▼
  Hyperparameter Tuning (Optuna)
  - XGBoost Tuned (30 trials)
  - LightGBM Tuned (30 trials)
  - Ridge Tuned (20 trials)
       │
       ▼
  Evaluasi Regresi
  - MSE, RMSE, MAE, R²
  - Actual vs Predicted plot
  - Residual analysis
       │
       ▼
  Model Interpretability (LIME)
  - Penjelasan prediksi per data point
       │
       ▼
  MLflow Experiment Tracking
```

---

## 🤖 Model yang Digunakan

| No | Model | Jenis |
|----|-------|-------|
| 1 | Linear Regression | Baseline linear |
| 2 | Ridge Regression | Regularized linear |
| 3 | Lasso Regression | Regularized linear |
| 4 | Decision Tree | Tree-based |
| 5 | Random Forest | Ensemble bagging |
| 6 | XGBoost | Gradient boosting |
| 7 | LightGBM | Fast gradient boosting |
| 8 | **XGBoost + Optuna** | Tuned — terbaik |
| 9 | LightGBM + Optuna | Tuned |
| 10 | Ridge + Optuna | Tuned |

---

## 📈 Metrik Evaluasi

| Metrik | Keterangan |
|--------|-----------|
| **MSE** | Mean Squared Error — rata-rata kuadrat error |
| **RMSE** | Root MSE — error dalam satuan tahun |
| **MAE** | Mean Absolute Error — rata-rata error absolut |
| **R²** | Koefisien determinasi — proporsi variansi yang dijelaskan model |

> Metrik utama: **RMSE** dan **R²** karena paling intuitif untuk regresi tahun

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost + Optuna | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| ... | ... | ... | ... |

*(Jalankan notebook untuk hasil aktual)*

---

## 🔍 LIME — Model Interpretability

LIME (Local Interpretable Model-agnostic Explanations) digunakan untuk menjelaskan **mengapa** model membuat prediksi tertentu pada level individual data point:

- Fitur dengan nilai positif (biru) → mendorong prediksi ke tahun **lebih baru**
- Fitur dengan nilai negatif (oranye) → mendorong prediksi ke tahun **lebih lama**

---

## 🛠️ Tools & Library

| Tool | Fungsi |
|------|--------|
| **Pandas, NumPy** | Manipulasi data |
| **Scikit-learn** | Preprocessing, model, evaluasi |
| **XGBoost** | Gradient boosting regressor |
| **LightGBM** | Fast gradient boosting |
| **Optuna** | Hyperparameter tuning otomatis |
| **LIME** | Model interpretability |
| **MLflow** | Experiment tracking & logging |
| **Matplotlib, Seaborn** | Visualisasi |

---

## ▶️ Cara Menjalankan

### 1. Upload dataset ke Google Drive
Letakkan `midterm-regresi-dataset.csv` di:
```
Google Drive → regresi_dataset/midterm-regresi-dataset.csv
```

### 2. Buka notebook di Google Colab
- Buka [colab.research.google.com](https://colab.research.google.com)
- Upload `regresi_song_year_colab.ipynb`

### 3. Aktifkan GPU
`Runtime` → `Change runtime type` → pilih **T4 GPU**

### 4. Run All
`Runtime` → `Run all` (`Ctrl + F9`)

### 5. Lihat MLflow (opsional, di local)
```bash
mlflow ui
```
Buka: http://localhost:5000

---

## 💡 Key Insights

- Prediksi tahun rilis lagu dari fitur audio adalah **non-trivial** — fitur audio tidak secara langsung merepresentasikan era
- **Boosting models** (XGBoost, LightGBM) jauh mengungguli model linear
- **RMSE** yang rendah berarti model bisa memprediksi tahun dengan error kecil
- **LIME** mengungkap fitur audio mana yang paling mempengaruhi prediksi era lagu
