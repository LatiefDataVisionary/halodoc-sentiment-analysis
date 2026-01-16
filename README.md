
# Sentiment Analysis Aplikasi Halodoc dengan IndoBERT

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21C?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 🌐 Demo Aplikasi  <-- TEMPEL DI SINI (BARU)

Ingin mencoba aplikasi ini secara langsung tanpa perlu instalasi?
Silakan kunjungi versi *live demo* yang telah di-deploy di Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/latief18/halodoc-sentiment-analysis)

> **Catatan:** Karena menggunakan *Free Tier* (CPU Basic), proses prediksi mungkin membutuhkan waktu beberapa detik.

## 📋 Ringkasan Proyek

Proyek ini bertujuan untuk membangun dan melatih model Analisis Sentimen berbasis ulasan pengguna aplikasi Halodoc dari Google Play Store. Model ini menggunakan arsitektur IndoBERT yang telah di-*fine-tune* untuk mengklasifikasikan sentimen ulasan ke dalam tiga kategori: **Negatif (0), Netral (1), dan Positif (2)**. Data ulasan diperoleh melalui *scraping* Google Play Store dan dataset eksternal, kemudian melalui serangkaian tahapan preprocessing (pembersihan, normalisasi slang) dan pelabelan otomatis menggunakan model IndoBERT pre-trained, sebelum akhirnya di-*fine-tune* untuk tugas klasifikasi ini. Aplikasi Streamlit akan digunakan untuk deployment interaktif.

## ✨ Fitur Utama

*   **Akuisisi Data Otomatis:** Mengumpulkan ulasan aplikasi Halodoc langsung dari Google Play Store.
*   **Preprocessing Teks Komprehensif:** Tahapan pembersihan data seperti *case folding*, penghapusan URL/angka/tanda baca, normalisasi *slang word*, dan penghapusan duplikasi.
*   **Pelabelan Sentimen Otomatis:** Menggunakan model IndoBERT pre-trained (`mdhugol/indonesia-bert-sentiment-classification`) untuk melabeli ulasan, memfasilitasi pembuatan dataset berlabel dalam skala besar.
*   **Exploratory Data Analysis (EDA):** Analisis mendalam terhadap distribusi sentimen, panjang kalimat, *word cloud*, dan N-gram untuk memahami karakteristik data.
*   **Fine-tuning IndoBERT:** Melatih ulang model IndoBERT (`indobenchmark/indobert-base-p1`) untuk klasifikasi sentimen 3-kelas.
*   **Evaluasi Model:** Menggunakan metrik seperti *Classification Report* dan *Confusion Matrix* untuk menilai performa model.
*   **Deployment Ready:** Model dan *tokenizer* disimpan dalam format yang siap untuk deployment.

## 🚀 Struktur Folder Deployment

Untuk deployment aplikasi Streamlit, struktur folder yang disiapkan adalah sebagai berikut:

```
. (root folder proyek)
├── app.py                           # Aplikasi utama Streamlit (UI & interaksi)
├── model_inference.py               # Modul berisi logika pemuatan model & prediksi
├── model_halodoc_sentiment/         # Folder berisi model dan tokenizer yang sudah di-fine-tune
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── requirements.txt                 # Daftar pustaka Python yang dibutuhkan
└── README.md                        # Dokumentasi proyek (file ini)
```

## 🛠️ Instalasi & Penggunaan

### Persyaratan Sistem

*   Python 3.8+
*   pip (manajer paket Python)
*   Lingkungan virtual (direkomendasikan)

### Langkah-langkah Instalasi

1.  **Clone repositori ini**:
    ```bash
    git clone https://github.com/LatiefDataVisionary/halodoc-sentiment-analysis/
    cd halodoc-sentiment-analysis
    ```

2.  **Buat dan aktifkan lingkungan virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    # atau
    venv\Scriptsctivate      # Untuk Windows
    ```

3.  **Instal semua dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **📥 UNDUH MODEL (PENTING!):**
    Karena ukuran model melebihi batas GitHub, Anda wajib mengunduhnya secara manual melalui link berikut:
    
    👉 **[Download Model IndoBERT (Google Drive)](https://drive.google.com/file/d/1kZo_oIQb5CCycfGzpGoJEzCgZ84VFptl/view)**

    **Instruksi:**
    *   Unduh file/folder dari link di atas.
    *   Jika berbentuk ZIP, ekstrak terlebih dahulu.
    *   Pastikan nama foldernya adalah `model_halodoc_sentiment`.
    *   Letakkan folder tersebut di dalam folder utama proyek ini (sejajar dengan `app.py`).

### Cara Menjalankan Aplikasi Streamlit

Setelah instalasi selesai dan lingkungan virtual aktif, Anda dapat menjalankan aplikasi Streamlit dengan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser web Anda (biasanya di `http://localhost:8501`).

## 🧠 Detail Model

Model yang digunakan untuk Analisis Sentimen adalah **IndoBERT Base P1** yang telah di-*fine-tune* pada dataset ulasan Halodoc yang sudah melalui preprocessing dan pelabelan otomatis. Model ini mampu mengklasifikasikan sentimen ke dalam 3 kategori: Negatif, Netral, dan Positif.

Proses pelatihan menggunakan *weighted cross-entropy loss* untuk menangani *imbalanced data*, *AdamW optimizer*, dan *linear scheduler with warmup*.

## 📊 Hasil dan Performa

Model mencapai akurasi validasi terbaik sekitar **0.9558** (95.58%). Namun, perlu dicatat bahwa model menunjukkan gejala *overfitting* ringan (nilai *train loss* yang sangat rendah dan *validation loss* yang lebih tinggi dan meningkat seiring *epoch*). Teknik *early stopping* (menyimpan model dengan akurasi validasi terbaik) telah diterapkan untuk mengatasi ini.

Berikut adalah ringkasan performa model pada *test set*:

```
              precision    recall  f1-score   support

     Negatif       0.90      0.93      0.92       780
      Netral       0.89      0.80      0.84       582
     Positif       0.97      0.98      0.98      3996

    accuracy                           0.96      5358
   macro avg       0.92      0.90      0.91      5358
weighted avg       0.96      0.96      0.96      5358

Akurasi Total: 0.9560
```

*Catatan: Untuk detail lengkap mengenai proses pelatihan, EDA, dan visualisasi, silakan lihat file Jupyter Notebook utama proyek ini.*

## 📧 Kontak

Jika Anda memiliki pertanyaan atau ingin berdiskusi lebih lanjut tentang proyek ini, silakan hubungi:

*   Lathif Ramadhan (datasciencelatief@gmail.com)
Atau bisa klik [disini](mailto:alamat@email.com)

---
