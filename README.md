#Midterm Machine Learning Project  
**Nama:** Naufal Alif Abyan  
**NIM:** 101032300032  

Repository ini berisi implementasi beberapa tugas Machine Learning yang dikerjakan menggunakan **Google Colab / Jupyter Notebook**.  
Seluruh project mencakup proses **data loading, preprocessing, training model, hyperparameter tuning, evaluasi model, interpretasi hasil, dan visualisasi**.

Project dalam repository ini terdiri dari 3 bagian utama:

1. **Fraud Detection Classification**  
   Memprediksi apakah suatu transaksi merupakan fraud atau bukan.

2. **Song Release Year Regression**  
   Memprediksi tahun rilis lagu berdasarkan fitur numerik audio.

3. **Customer Segmentation Clustering**  
   Mengelompokkan pelanggan berdasarkan perilaku transaksi menggunakan clustering.

---

# Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Project 1 - Fraud Detection Classification](#project-1---fraud-detection-classification)
- [Project 2 - Song Release Year Regression](#project-2---song-release-year-regression)
- [Project 3 - Customer Segmentation Clustering](#project-3---customer-segmentation-clustering)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Author](#author)

---

# Project Overview

Repository ini dibuat untuk menunjukkan implementasi **end-to-end machine learning workflow** pada beberapa jenis permasalahan:

- **Classification** → menentukan label fraud / non-fraud  
- **Regression** → memprediksi nilai kontinu berupa tahun rilis lagu  
- **Clustering** → mengelompokkan data pelanggan ke dalam beberapa cluster  

Setiap project mencakup tahapan berikut:

- Data loading
- Data preprocessing
- Missing value handling
- Outlier handling
- Feature engineering / feature preparation
- Train-validation split
- Model training
- Hyperparameter tuning dengan **Optuna**
- Evaluasi model
- Visualisasi hasil
- Interpretasi model (LIME pada regression)
- Experiment tracking dengan **MLflow** (pada classification dan regression)

---

# Repository Structure

Struktur repository yang disarankan:

```bash
machine-learning-project/
│
├── notebooks/
│   ├── transaction_classification.ipynb
│   ├── song_regression.ipynb
│   └── clustering.ipynb
│
├── data/
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── midterm-regresi-dataset.csv
│   └── clusteringmidterm.csv
│
├── outputs/
│   ├── submission_fraud.csv
│   ├── regression_results.csv
│   └── clustering_analysis.csv
│
├── images/
│   ├── confusion_matrix.png
│   ├── actual_vs_predicted.png
│   ├── elbow_method.png
│   └── pca_cluster_visualization.png
│
├── README.md
└── requirements.txt
