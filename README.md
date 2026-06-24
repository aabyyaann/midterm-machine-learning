# Midterm Machine Learning Project

**Nama:** Naufal Alif Abyan
**NIM:** 101032300032

Repository ini berisi implementasi tugas **Midterm Machine Learning** yang mencakup tiga jenis permasalahan utama dalam machine learning, yaitu:

1. **Classification** – Mendeteksi transaksi fraud pada data transaksi online
2. **Regression** – Memprediksi tahun rilis lagu berdasarkan fitur audio numerik
3. **Clustering** – Mengelompokkan pelanggan berdasarkan perilaku transaksi dan karakteristik finansial

Seluruh project dikerjakan menggunakan **Python** di **Google Colab / Jupyter Notebook** dengan pendekatan **end-to-end machine learning workflow**, mulai dari data loading, preprocessing, model training, hyperparameter tuning, evaluasi model, hingga visualisasi hasil.

---

# Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Project 1 - Fraud Detection Classification](#project-1---fraud-detection-classification)
* [Project 2 - Song Release Year Regression](#project-2---song-release-year-regression)
* [Project 3 - Customer Segmentation Clustering](#project-3---customer-segmentation-clustering)
* [How to Run](#how-to-run)
* [Requirements](#requirements)
* [Author](#author)

---

# Project Overview

Project Midterm ini dibuat untuk menunjukkan penerapan beberapa teknik machine learning pada tiga tipe masalah yang berbeda:

* **Classification**
  Digunakan untuk memprediksi apakah suatu transaksi termasuk fraud atau tidak.

* **Regression**
  Digunakan untuk memprediksi nilai kontinu berupa **tahun rilis lagu** dari fitur numerik audio.

* **Clustering**
  Digunakan untuk mengelompokkan pelanggan ke dalam beberapa segmen berdasarkan perilaku finansial mereka.

Setiap notebook pada repository ini menerapkan tahapan machine learning yang relatif lengkap, yaitu:

* Data loading
* Data preprocessing
* Missing value handling
* Outlier handling
* Feature preparation
* Train-validation split
* Model training
* Hyperparameter tuning menggunakan **Optuna**
* Evaluasi model
* Visualisasi hasil
* Interpretasi model (LIME pada regression)
* Experiment tracking menggunakan **MLflow** (classification dan regression)

---

# Repository Structure

Struktur repository yang digunakan / disarankan:

```bash
midterm-machine-learning/
│
├── Transaction.ipynb
├── Regresi.ipynb
├── Clustering.ipynb
│
├── README.md
└── requirements.txt
```

Jika dataset tidak diunggah ke GitHub karena ukuran file besar, maka notebook dapat membaca dataset langsung dari **Google Drive**.

Contoh lokasi dataset pada Google Drive:

```python
PATH = "/content/drive/MyDrive/Midterm ML/"
```

---

# Project 1 - Fraud Detection Classification

## Objective

Project ini bertujuan untuk membangun model **classification** yang dapat mendeteksi apakah suatu transaksi online merupakan **fraud** atau **non-fraud**.

Target variabel:

* **`isFraud`**

---

## Dataset

Dataset yang digunakan pada project classification:

* **train_transaction.csv**
* **test_transaction.csv**

Penjelasan:

* `train_transaction.csv` digunakan sebagai data training dan validation
* `test_transaction.csv` digunakan untuk menghasilkan prediksi akhir / submission

Dataset transaksi berisi berbagai informasi terkait transaksi online, seperti:

* jumlah transaksi (`TransactionAmt`)
* kode produk (`ProductCD`)
* informasi kartu (`card1`, `card2`, dan seterusnya)
* alamat (`addr1`, `addr2`)
* serta berbagai fitur numerik lainnya

---

## Workflow

Tahapan yang dilakukan pada notebook classification:

### 1. Data Loading

Dataset dibaca menggunakan `pandas.read_csv()` dari Google Drive.

### 2. Memory Optimization

Untuk mengurangi penggunaan memori, tipe data numerik diubah:

* `float64` → `float32`
* `int64` → `int32`

### 3. Feature Definition

Seluruh kolom digunakan sebagai fitur kecuali:

* `TransactionID`
* `isFraud`

### 4. Missing Value Handling

* Kolom kategorikal diisi dengan nilai `"Unknown"`
* Kolom numerik diisi menggunakan **median**

### 5. Encoding Categorical Features

Kolom kategorikal diubah menjadi numerik menggunakan **LabelEncoder**.
Agar aman terhadap kategori baru di data test, encoder dilatih pada gabungan data train dan test.

### 6. Train-Validation Split

Data dibagi menjadi:

* **training set**
* **validation set**

menggunakan `train_test_split()` dengan `stratify=y`.

### 7. Hyperparameter Tuning

Hyperparameter model dioptimasi menggunakan **Optuna**.
Parameter yang dicari meliputi:

* `learning_rate`
* `num_leaves`

### 8. Model Training

Model utama yang digunakan:

* **LightGBM Classifier**

### 9. Model Evaluation

Evaluasi dilakukan menggunakan:

* **ROC-AUC**
* **Confusion Matrix**
* **Classification Report**

### 10. Submission Generation

Model terbaik digunakan untuk memprediksi data test, kemudian hasilnya disimpan sebagai:

* **`submission_fraud.csv`**

---

## Model Used

Model yang digunakan pada project classification:

* **LightGBM Classifier**
* Hyperparameter tuning dengan **Optuna**

---

## Evaluation Metric

Karena fraud detection merupakan **imbalanced classification problem**, metrik evaluasi yang digunakan adalah:

* **ROC-AUC**
  Mengukur kemampuan model membedakan fraud dan non-fraud

* **Precision**
  Mengukur seberapa banyak prediksi fraud yang benar-benar fraud

* **Recall**
  Mengukur seberapa banyak fraud yang berhasil dideteksi model

* **F1-Score**
  Harmonic mean antara precision dan recall

* **Confusion Matrix**
  Menunjukkan jumlah prediksi benar dan salah untuk masing-masing kelas

---

## Output

Output utama dari project classification:

* File prediksi:

  * **`submission_fraud.csv`**
* Visualisasi:

  * **Confusion Matrix**
* Skor evaluasi:

  * **ROC-AUC**
  * **Classification Report**

---

# Project 2 - Song Release Year Regression

## Objective

Project ini bertujuan untuk membangun model **regression** untuk memprediksi **tahun rilis lagu** berdasarkan fitur numerik audio.

Target variabel:

* **Kolom pertama pada dataset** (tahun rilis lagu)

---

## Dataset

Dataset yang digunakan:

* **midterm-regresi-dataset.csv**

Karakteristik dataset:

* Kolom pertama = **target** (tahun rilis lagu)
* Kolom selanjutnya = **fitur numerik audio**

Karena nama fitur asli tidak tersedia, seluruh kolom fitur diberi nama ulang menjadi:

* `feature_1`
* `feature_2`
* `feature_3`
* dan seterusnya

---

## Workflow

Tahapan pada notebook regression:

### 1. Data Loading

Dataset dibaca menggunakan `pandas.read_csv()` dengan `header=None`.

### 2. Feature & Target Separation

* `y` = kolom pertama (target tahun)
* `X` = seluruh kolom setelah target

### 3. Feature Renaming

Fitur diubah menjadi nama yang lebih mudah dibaca:

* `feature_1`
* `feature_2`
* ...
* `feature_n`

### 4. Outlier Handling

Outlier pada fitur ditangani menggunakan **clipping** berdasarkan quantile:

* lower bound = **1% quantile**
* upper bound = **99% quantile**

Tujuannya adalah mengurangi pengaruh nilai ekstrem pada model.

### 5. Feature Scaling

Seluruh fitur dinormalisasi menggunakan:

* **StandardScaler**

### 6. Train-Validation Split

Data dibagi menjadi:

* **training set**
* **validation set**

menggunakan `train_test_split()`.

### 7. Hyperparameter Tuning

Model dituning menggunakan **Optuna** dengan parameter seperti:

* `learning_rate`
* `num_leaves`

### 8. Model Training

Model utama yang digunakan:

* **LightGBM Regressor**

### 9. Model Evaluation

Evaluasi model dilakukan menggunakan:

* **MSE**
* **RMSE**
* **MAE**
* **R² Score**

### 10. Model Interpretation

Untuk memahami prediksi model, digunakan **LIME (Local Interpretable Model-agnostic Explanations)** pada salah satu sampel data validasi.

### 11. Visualization

Prediksi model divisualisasikan menggunakan scatter plot:

* **Actual vs Predicted Year**

---

## Model Used

Model yang digunakan pada project regression:

* **LightGBM Regressor**
* Hyperparameter tuning dengan **Optuna**
* Interpretasi model menggunakan **LIME**

---

## Evaluation Metric

Metrik evaluasi yang digunakan pada regression:

* **MSE (Mean Squared Error)**
  Mengukur rata-rata error kuadrat antara nilai aktual dan prediksi

* **RMSE (Root Mean Squared Error)**
  Menunjukkan error prediksi dalam satuan tahun

* **MAE (Mean Absolute Error)**
  Mengukur rata-rata error absolut

* **R² Score**
  Mengukur seberapa baik model menjelaskan variasi target

---

## Example Result

Contoh hasil evaluasi dari notebook regression:

* **RMSE**: sekitar **9.0453**
* **MAE**: sekitar **6.3243**
* **R² Score**: sekitar **0.3125**

Hasil tersebut menunjukkan bahwa model mampu memprediksi tahun rilis lagu dengan tingkat error yang masih cukup wajar, meskipun performa model masih dapat ditingkatkan lebih lanjut.

---

## Output

Output utama dari project regression:

* Nilai evaluasi:

  * MSE
  * RMSE
  * MAE
  * R² Score

* Visualisasi:

  * **Actual vs Predicted Year**

* Interpretasi:

  * **LIME explanation** untuk salah satu sampel data validasi

---

# Project 3 - Customer Segmentation Clustering

## Objective

Project ini bertujuan untuk melakukan **customer segmentation** menggunakan teknik **clustering**.
Tujuannya adalah mengelompokkan pelanggan berdasarkan karakteristik finansial dan perilaku transaksi sehingga tiap kelompok pelanggan dapat dianalisis lebih lanjut.

---

## Dataset

Dataset yang digunakan:

* **clusteringmidterm.csv**

Salah satu kolom dalam dataset adalah:

* **`CUST_ID`**

Kolom ini dihapus pada proses preprocessing karena hanya berfungsi sebagai identitas unik pelanggan dan tidak relevan sebagai fitur clustering.

---

## Workflow

Tahapan pada notebook clustering:

### 1. Data Loading

Dataset dibaca menggunakan `pandas.read_csv()`.

### 2. Remove Identifier

Kolom `CUST_ID` dihapus dari data karena tidak digunakan sebagai fitur.

### 3. Missing Value Handling

Nilai kosong diisi menggunakan **median**.

### 4. Outlier Handling

Outlier ditangani menggunakan **clipping**:

* lower bound = **5% quantile**
* upper bound = **95% quantile**

### 5. Feature Scaling

Seluruh fitur diskalakan menggunakan:

* **StandardScaler**

Scaling diperlukan karena **K-Means sensitif terhadap skala data**.

### 6. Elbow Method

Jumlah cluster optimal ditentukan menggunakan **Elbow Method** dengan membandingkan nilai inertia untuk beberapa nilai `k`.

### 7. Clustering Model

Model clustering yang digunakan:

* **KMeans**

Pada implementasi notebook, jumlah cluster akhir yang dipilih adalah:

* **k = 4**

### 8. Model Evaluation

Evaluasi dilakukan menggunakan:

* **Silhouette Score**

### 9. Cluster Visualization

Hasil clustering divisualisasikan dalam 2 dimensi menggunakan:

* **PCA (Principal Component Analysis)**

### 10. Cluster Analysis

Setelah cluster terbentuk, dilakukan analisis rata-rata beberapa fitur utama, seperti:

* `BALANCE`
* `PURCHASES`
* `CASH_ADVANCE`
* `CREDIT_LIMIT`
* `PAYMENTS`

Analisis ini digunakan untuk memahami karakteristik masing-masing kelompok pelanggan.

---

## Model Used

Model yang digunakan pada project clustering:

* **K-Means Clustering**
* **PCA** untuk visualisasi
* **Silhouette Score** untuk evaluasi kualitas cluster

---

## Evaluation Metric

Metrik evaluasi utama pada clustering adalah:

* **Silhouette Score**

Interpretasi:

* mendekati **1** → cluster terpisah dengan baik
* mendekati **0** → cluster saling tumpang tindih
* bernilai negatif → hasil clustering kurang baik

---

## Example Result

Contoh hasil evaluasi dari notebook clustering:

* **Silhouette Score**: sekitar **0.1934**

Nilai ini menunjukkan bahwa cluster yang terbentuk belum terlalu terpisah kuat, namun masih dapat digunakan untuk analisis segmentasi pelanggan secara umum.

---

## Output

Output utama dari project clustering:

* Grafik **Elbow Method**
* Visualisasi cluster menggunakan **PCA**
* Tabel rata-rata karakteristik tiap cluster
* Kesimpulan singkat karakteristik masing-masing cluster pelanggan

---

# How to Run

## 1. Clone Repository

Clone repository ini ke local machine:

```bash
git clone https://github.com/USERNAME/NAMA-REPOSITORY.git
cd NAMA-REPOSITORY
```

Ganti `USERNAME/NAMA-REPOSITORY` dengan alamat repository GitHub milikmu.

---

## 2. Install Required Libraries

Install seluruh library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Jika menjalankan di Google Colab, library juga bisa diinstall langsung di notebook:

```python
!pip install mlflow optuna lightgbm lime -q
```

---

## 3. Siapkan Dataset

Pastikan seluruh dataset tersedia pada folder kerja atau Google Drive.

Dataset yang digunakan:

* `train_transaction.csv`
* `test_transaction.csv`
* `midterm-regresi-dataset.csv`
* `clusteringmidterm.csv`

Contoh path pada Google Colab:

```python
PATH = "/content/drive/MyDrive/Midterm ML/"
```

---

## 4. Jalankan Notebook

Jalankan notebook sesuai project yang ingin dibuka:

1. **Transaction.ipynb**
2. **Regresi.ipynb**
3. **Clustering.ipynb**

Pastikan seluruh cell dijalankan berurutan dari atas ke bawah.

---

# Requirements

Library utama yang digunakan pada project ini:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `lightgbm`
* `optuna`
* `mlflow`
* `lime`

Contoh isi file `requirements.txt`:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
optuna
mlflow
lime
```

---

# Conclusion

Repository **Midterm Machine Learning** ini mencakup tiga implementasi utama machine learning:

1. **Fraud Detection Classification**
   Menggunakan LightGBM Classifier untuk memprediksi transaksi fraud pada data transaksi online.

2. **Song Release Year Regression**
   Menggunakan LightGBM Regressor untuk memprediksi tahun rilis lagu berdasarkan fitur numerik audio, disertai interpretasi model menggunakan LIME.

3. **Customer Segmentation Clustering**
   Menggunakan K-Means untuk mengelompokkan pelanggan berdasarkan perilaku finansial dan transaksi.

Secara keseluruhan, project ini menunjukkan penerapan workflow machine learning yang cukup lengkap, mulai dari preprocessing hingga evaluasi dan interpretasi model.

---

# Author

**Naufal Alif Abyan**
**NIM:** 101032300032
**Course:** Machine Learning
