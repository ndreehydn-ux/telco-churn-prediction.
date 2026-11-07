# 📉 Telco Customer Churn Prediction

Prediksi pelanggan telekomunikasi yang berisiko berhenti berlangganan (*churn*), lengkap dengan analisis faktor penyebab dan simulasi dampak bisnis.

> Project kelompok mata kuliah Data Modeling, Program Studi Sistem Informasi, Universitas Multimedia Nusantara (2026). Peran saya: **Project Lead**, yang mencakup arsitektur pipeline, evaluasi, dan revisi metodologi.

## 🎯 Masalah Bisnis
Mempertahankan pelanggan jauh lebih murah dibanding mencari pelanggan baru. Tim retensi butuh jawaban atas dua pertanyaan: **siapa** pelanggan yang paling berisiko churn, dan **mengapa**.

## 📊 Hasil Utama
| Metrik (data uji, 118.839 pelanggan) | Nilai |
|---|---|
| PR-AUC | **0,754** (baseline acak: 0,225) |
| ROC-AUC | **0,916** |
| Recall kelas churn (threshold 0,33) | **80,5%** |
| Churner tertangkap dengan menghubungi top 20% risiko | **63,5%** (lift 3,2×) |

![Confusion matrix, ROC, PR curve](images/confusion_roc_pr.png)

## 🔍 Faktor Penyebab Churn (SHAP)
- **Tenure rendah** (pelanggan baru) adalah faktor risiko terbesar.
- Kontrak **month-to-month** menaikkan risiko, sedangkan kontrak **dua tahun** menjadi faktor pelindung terkuat.
- Pembayaran via **electronic check**, **paperless billing**, dan **fiber optic** menaikkan risiko.
- Pelanggan **tanpa Online Security / Tech Support** lebih mudah pergi.

![SHAP importance](images/shap_importance.png)

## 🧪 Metodologi
1. **EDA**: 594.194 pelanggan, 19 fitur (16 kategorikal), 22,5% churn.
2. **Split dulu, baru preprocessing**: 80/20 stratified, sehingga data uji mempertahankan distribusi asli.
3. **Pipeline bebas data leakage**: One-Hot Encoding, scaling, dan penanganan imbalance berada di dalam pipeline.
4. **Perbandingan 18 konfigurasi**: 6 algoritma (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) + baseline Dummy × 3 strategi imbalance (none, class weight, SMOTE), dengan 5-fold CV dan metrik **PR-AUC**.
5. **Tuning** model terbaik (CatBoost) menggunakan RandomizedSearchCV.
6. **Threshold** dipilih dari prediksi out-of-fold data latih, tidak menyentuh data uji.
7. **Interpretasi & bisnis**: SHAP, cumulative gains, dan simulasi profit program retensi.

![Perbandingan model](images/model_comparison.png)

## 💡 Temuan Menarik
- **Model boosting unggul**, tetapi CatBoost, LightGBM, dan XGBoost praktis setara (selisih < std antar fold). LightGBM ~55× lebih cepat dilatih.
- **SMOTE tidak pernah menjadi strategi terbaik.** Ketika diterapkan dengan benar (hanya pada data latih), SMOTE tidak meningkatkan PR-AUC. Ketidakseimbangan kelas lebih efektif ditangani dengan threshold tuning.
- **Koreksi data leakage**: versi awal project menerapkan SMOTE sebelum split, sehingga accuracy tampak 88,9%. Setelah diperbaiki, accuracy menjadi 86,1%. Angkanya lebih rendah, tetapi jujur.

## 🗂️ Struktur Repo
| File | Isi |
|---|---|
| `churn_prediction.ipynb` | Notebook lengkap: EDA → modeling → evaluasi → SHAP |
| `reports/` | Laporan akhir (PDF) |
| `images/` | Grafik hasil |

## ▶️ Cara Menjalankan
1. Unduh `train.csv` dari [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) dan letakkan di folder project.
2. `pip install -r requirements.txt`
3. Jalankan `churn_prediction.ipynb`.

## 🛠️ Tools
Python · pandas · scikit-learn · imbalanced-learn · XGBoost · LightGBM · CatBoost · SHAP

## 👥 Tim
Andree Hayden Fernando (Project Lead) · Frans Marco Haryanto (Model Developer) · Kenzie Tristan Tang (Presenter) · Kingsty James Wayne (Data Analyst)