# 🔍 Fraud Detection — End-to-End Machine Learning Pipeline

> **UTS Machine Learning — Individual Task**
> "Hands-On End-to-End Models Machine Learning"

---

## 📌 Identitas

| Field | Detail |
|-------|--------|
| **Nama** | Syeh Khatami |
| **NIM** | 101032300157 |
| **Kelas** | Machine Learning |

---

## 🎯 Tujuan Project

Membangun pipeline Machine Learning end-to-end untuk mendeteksi transaksi online yang bersifat fraudulent (penipuan) menggunakan dataset IEEE-CIS Fraud Detection.

Target prediksi: **`isFraud`** — probabilitas bahwa sebuah transaksi adalah penipuan (1 = Fraud, 0 = Non-Fraud).

---

## 📂 Struktur Repository

```
midterm-machine-learning/
│
├── fraud_detection_ml.ipynb   ← Notebook utama (semua kode & hasil)
├── model_results.csv          ← Tabel perbandingan semua model
├── class_distribution.png     ← Visualisasi distribusi kelas
├── model_comparison.png       ← Perbandingan metrik semua model
├── confusion_matrix.png       ← Confusion matrix best model
├── roc_curve.png              ← ROC Curve best model
├── feature_importance.png     ← Feature importance XGBoost
├── optuna_history.png         ← Optuna optimization history
└── README.md                  ← File ini
```

> ⚠️ **Dataset tidak disertakan** karena ukurannya ~600MB.
> Lihat bagian **Dataset** di bawah untuk cara mendapatkannya.

---

## 📊 Dataset

**Nama:** IEEE-CIS Fraud Detection Dataset
**Sumber:** [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

### File yang Digunakan:
| File | Keterangan |
|------|-----------|
| `train_transaction.csv` | Data transaksi (fitur amount, waktu, kartu, dll) |
| `train_identity.csv` | Data identitas per transaksi (device, browser, dll) |

### Cara Download:
1. Buat akun [Kaggle](https://kaggle.com) jika belum punya
2. Masuk ke kompetisi: https://www.kaggle.com/c/ieee-fraud-detection
3. Klik **Download All** atau gunakan Kaggle CLI:
   ```bash
   kaggle competitions download -c ieee-fraud-detection
   ```
4. Letakkan `train_transaction.csv` dan `train_identity.csv` di folder yang sama dengan notebook

---

## 🔧 Alur Pipeline

```
Raw Data (train_transaction + train_identity)
       │
       ▼
  Data Merging (join by TransactionID)
       │
       ▼
  Exploratory Data Analysis (EDA)
  - Distribusi kelas (isFraud ~3.5%)
  - Missing values analysis
  - Distribusi TransactionAmt
       │
       ▼
  Preprocessing & Feature Engineering
  - Drop kolom missing > 80%
  - Label Encoding (kategorikal)
  - Median Imputation
  - Log transform TransactionAmt
  - Tambah fitur: TransactionHour, TransactionDay
       │
       ▼
  Train-Test Split (80:20, stratified)
       │
       ▼
  Model Training (5 Baseline Models)
  ┌─────────────────────────────┐
  │ 1. Logistic Regression      │
  │ 2. Decision Tree            │
  │ 3. Random Forest            │
  │ 4. XGBoost                  │
  │ 5. LightGBM                 │
  └─────────────────────────────┘
       │
       ▼
  Hyperparameter Tuning (Optuna)
  - XGBoost Tuned (30 trials)
  - LightGBM Tuned (30 trials)
       │
       ▼
  Evaluasi & Perbandingan
  - ROC-AUC, F1, Precision, Recall, Accuracy
       │
       ▼
  MLflow Experiment Tracking
```

---

## 🤖 Model yang Digunakan

| No | Model | Keterangan |
|----|-------|-----------|
| 1 | Logistic Regression | Baseline linear model |
| 2 | Decision Tree | Baseline tree model |
| 3 | Random Forest | Ensemble bagging |
| 4 | XGBoost | Gradient boosting (baseline) |
| 5 | LightGBM | Gradient boosting (baseline) |
| 6 | **XGBoost + Optuna** | Tuned — **best model** |
| 7 | LightGBM + Optuna | Tuned |

---

## 📈 Hasil & Metrik

> Metrik utama: **ROC-AUC** karena dataset sangat imbalanced (~3.5% fraud)

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost + Optuna | *lihat notebook* | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| LightGBM + Optuna | *lihat notebook* | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| Random Forest | *lihat notebook* | *lihat notebook* | *lihat notebook* | *lihat notebook* |
| ... | ... | ... | ... | ... |

*(Jalankan notebook untuk mendapatkan hasil aktual)*

---

## 🛠️ Tools & Library

| Tool | Fungsi |
|------|--------|
| **Pandas, NumPy** | Manipulasi data |
| **Scikit-learn** | Preprocessing, model, evaluasi |
| **XGBoost** | Gradient boosting |
| **LightGBM** | Fast gradient boosting |
| **Optuna** | Hyperparameter tuning otomatis |
| **MLflow** | Experiment tracking & logging |
| **Matplotlib, Seaborn** | Visualisasi |

---

## ▶️ Cara Menjalankan Notebook

### 1. Install dependencies
```bash
pip install optuna mlflow xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
```

### 2. Siapkan dataset
Letakkan `train_transaction.csv` dan `train_identity.csv` di folder yang sama dengan notebook.

### 3. Jalankan notebook
```bash
jupyter notebook fraud_detection_ml.ipynb
```
Jalankan sel dari atas ke bawah secara berurutan.

### 4. Lihat MLflow Dashboard (opsional)
```bash
mlflow ui
```
Buka browser: http://localhost:5000

---

## 💡 Key Insights

- **Class imbalance** sangat ekstrem (~3.5% fraud) → digunakan `scale_pos_weight` (XGBoost) dan `class_weight='balanced'`
- **Accuracy** bukan metrik yang tepat untuk kasus ini → fokus pada **ROC-AUC** dan **F1-Score**
- **Boosting models** (XGBoost, LightGBM) secara konsisten outperform model lain pada data tabular kompleks
- **Optuna** berhasil meningkatkan ROC-AUC dibanding baseline
- Fitur paling informatif: `TransactionAmt`, fitur `V` (Vesta-engineered), dan `card` features
