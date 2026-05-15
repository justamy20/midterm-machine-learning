# 💳 Customer Credit Card Clustering — End-to-End ML Pipeline

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

Membangun pipeline Machine Learning end-to-end untuk mengelompokkan pelanggan kartu kredit berdasarkan **perilaku penggunaan dan pembayaran** menggunakan metode unsupervised learning (clustering).

---

## 📂 Struktur Repository

```
midterm-machine-learning/
│
├── clustering_customer_colab.ipynb  ← Notebook utama
├── clustering_results.csv           ← Perbandingan semua model
├── cluster_profiles.csv             ← Profil rata-rata tiap cluster
├── data_with_clusters.csv           ← Data asli + label cluster
├── feature_distributions.png
├── correlation_heatmap.png
├── boxplot_outliers.png
├── scatter_eda.png
├── optimal_k.png                    ← Elbow + Silhouette + DB + CH
├── silhouette_plot.png
├── dendrogram.png
├── kdistance_plot.png               ← Untuk menentukan eps DBSCAN
├── model_comparison.png
├── pca_clusters.png                 ← Visualisasi PCA 2D
├── tsne_clusters.png                ← Visualisasi t-SNE 2D
├── radar_profile.png                ← Radar chart profil cluster
├── cluster_boxplots.png
└── README.md
```

> ⚠️ **Dataset tidak disertakan** — disediakan oleh dosen (clusteringmidterm.csv)

---

## 📊 Dataset

**File:** `clusteringmidterm.csv`

| Kolom | Deskripsi |
|-------|-----------|
| CUST_ID | ID unik pelanggan (di-drop saat modeling) |
| BALANCE | Saldo outstanding rata-rata |
| BALANCE_FREQUENCY | Seberapa sering balance diperbarui |
| PURCHASES | Total pembelian menggunakan kartu |
| ONEOFF_PURCHASES | Pembelian satu kali (besar) |
| INSTALLMENTS_PURCHASES | Pembelian cicilan |
| CASH_ADVANCE | Total tarik tunai |
| PURCHASES_FREQUENCY | Frekuensi pembelian umum |
| ONEOFF_PURCHASES_FREQUENCY | Frekuensi pembelian one-off |
| PURCHASES_INSTALLMENTS_FREQUENCY | Frekuensi pembelian cicilan |
| CASH_ADVANCE_FREQUENCY | Frekuensi tarik tunai |
| CASH_ADVANCE_TRX | Jumlah transaksi cash advance |
| PURCHASES_TRX | Jumlah transaksi pembelian |
| CREDIT_LIMIT | Limit kredit pelanggan |
| PAYMENTS | Total pembayaran |
| MINIMUM_PAYMENTS | Total minimum payment |
| PRC_FULL_PAYMENT | Proporsi bulan bayar penuh |
| TENURE | Lama kepemilikan kartu (bulan) |

---

## 🔧 Alur Pipeline

```
clusteringmidterm.csv
       │
       ▼
  Drop CUST_ID (kolom ID)
       │
       ▼
  EDA
  - Distribusi fitur, heatmap korelasi
  - Boxplot outlier detection
       │
       ▼
  Preprocessing
  - Median Imputation
  - Outlier clip (IQR)
  - Feature Engineering:
    PAYMENT_RATIO, CASH_ADVANCE_RATIO
    CREDIT_UTILIZATION, AVG_PURCHASE_PER_TRX
  - StandardScaler
       │
       ▼
  Penentuan K Optimal
  - Elbow Method (Inertia)
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Dendrogram (Hierarchical)
       │
       ▼
  Training Model Clustering
  ┌─────────────────────────────────┐
  │ K-Means (K optimal, K±1)        │
  │ Hierarchical (ward/complete/avg)│
  │ DBSCAN (eps dari k-distance)    │
  └─────────────────────────────────┘
       │
       ▼
  Evaluasi
  - Silhouette Score
  - Davies-Bouldin Score
  - Calinski-Harabasz Score
       │
       ▼
  Visualisasi & Interpretasi
  - PCA 2D, t-SNE 2D
  - Radar Chart profil cluster
  - Boxplot per cluster
  - Deskripsi tiap segmen pelanggan
       │
       ▼
  MLflow Tracking
```

---

## 🤖 Model yang Digunakan

| Model | Keterangan |
|-------|-----------|
| K-Means (K optimal) | Partisi berbasis centroid |
| K-Means (K-1, K+1) | Variasi K untuk perbandingan |
| Hierarchical - Ward | Linkage berdasarkan variansi |
| Hierarchical - Complete | Linkage berdasarkan jarak max |
| Hierarchical - Average | Linkage berdasarkan rata-rata |
| DBSCAN | Density-based, deteksi noise |

---

## 📈 Metrik Evaluasi Clustering

| Metrik | Keterangan |
|--------|-----------|
| **Silhouette Score** | -1 s.d. 1 — semakin tinggi semakin baik |
| **Davies-Bouldin** | ≥ 0 — semakin rendah semakin baik |
| **Calinski-Harabasz** | Semakin tinggi semakin baik |

---

## 👥 Profil Cluster (Contoh)

| Tipe Pelanggan | Karakteristik |
|----------------|--------------|
| 🛍️ Active Spender | Banyak belanja, jarang cash advance |
| 💸 Cash Advance User | Banyak tarik tunai, balance tinggi |
| ✅ Responsible User | Balance rendah, sering bayar penuh |
| 💎 High Credit Customer | Limit kredit tinggi |
| 😴 Low Activity | Jarang transaksi |

*(Profil aktual tergantung hasil clustering — lihat notebook)*

---

## 🛠️ Tools & Library

| Tool | Fungsi |
|------|--------|
| Pandas, NumPy | Manipulasi data |
| Scikit-learn | Clustering, evaluasi, PCA |
| SciPy | Hierarchical clustering, dendrogram |
| MLflow | Experiment tracking |
| Matplotlib, Seaborn | Visualisasi |
| Yellowbrick | Visualisasi ML tambahan |

---

## ▶️ Cara Menjalankan

1. Upload `clusteringmidterm.csv` ke Google Drive → folder `clustering_dataset/`
2. Buka notebook di [Google Colab](https://colab.research.google.com)
3. Runtime → Change runtime type → **T4 GPU**
4. Isi Nama & NIM di sel pertama
5. Runtime → **Run all** (`Ctrl+F9`)
6. File → **Save a copy in GitHub**
