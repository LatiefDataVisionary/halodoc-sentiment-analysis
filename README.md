# Sentiment Analysis Aplikasi Halodoc dengan IndoBERT

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21C?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/github/license/LatiefDataVisionary/halodoc-sentiment-analysis?style=for-the-badge)

## 📋 Ringkasan Proyek

**Halodoc Sentiment Analysis** adalah proyek *End-to-End Machine Learning* yang bertujuan untuk mengevaluasi kepuasan pengguna aplikasi *telemedicine* Halodoc. Proyek ini tidak hanya sekadar melakukan klasifikasi teks, tetapi juga menerapkan pendekatan **Data Science** yang komprehensif mulai dari akuisisi data, pembersihan tingkat lanjut, hingga *deployment* aplikasi berbasis *cloud*.

Inti dari proyek ini adalah penggunaan model **IndoBERT** (*Indonesian BERT*) yang telah di-*fine-tune* untuk memahami konteks bahasa Indonesia (termasuk bahasa gaul/slang) dalam ulasan aplikasi. Model ini mengklasifikasikan sentimen ke dalam tiga kategori: **Negatif (0), Netral (1), dan Positif (2)**.

Solusi ini hadir untuk mengatasi bias pada *rating* bintang di Google Play Store, di mana seringkali rating yang diberikan pengguna tidak selaras dengan isi ulasan teksnya.

## 🌐 Demo Aplikasi (Live)

Ingin mencoba aplikasi ini secara langsung tanpa instalasi? Kunjungi versi *live demo* yang telah di-deploy di Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/latief18/halodoc-sentiment-analysis)

> **Catatan:** Karena menggunakan infrastruktur *Free Tier* (CPU Basic), proses prediksi mungkin membutuhkan waktu beberapa detik.

## ✨ Fitur Utama

Aplikasi ini dirancang dengan antarmuka modern (*Glassmorphism UI*) dan memiliki fitur-fitur unggulan sebagai berikut:

### 1. 📊 Dashboard Analisis Interaktif
*   **Monitoring Real-time:** Menampilkan total ulasan dan distribusi sentimen dari dataset historis.
*   **Visualisasi Mendalam:** Dilengkapi dengan *Donut Chart* untuk proporsi sentimen, *Word Cloud* untuk melihat kata kunci dominan, dan analisis *N-Gram* (frasa).
*   **Data Explorer:** Tabel interaktif untuk meninjau data ulasan mentah beserta label hasil prediksi.

### 2. 📂 Analisis File Massal (*Batch Analysis*)
*   **Upload Fleksibel:** Mendukung unggah file eksternal (format `.csv` atau `.xlsx`) berisi ribuan ulasan baru.
*   **Otomatisasi Penuh:** Sistem melakukan *preprocessing* (pembersihan teks, normalisasi slang) dan prediksi secara otomatis.
*   **Instant Report:** Menghasilkan dashboard visualisasi khusus untuk data yang baru diunggah.
*   **Download Hasil:** Pengguna dapat mengunduh hasil analisis lengkap beserta *Confidence Score* dalam format CSV.

### 3. 🧪 Uji Coba Model (*Live Prediction*)
*   **Simulasi Input:** Pengguna dapat mengetikkan kalimat ulasan secara manual untuk menguji respon model.
*   **Transparansi Model:** Menampilkan hasil prediksi disertai tingkat keyakinan (*Confidence Score*) dan probabilitas untuk setiap kelas sentimen.

## 🚀 Struktur Direktori

Berikut adalah struktur folder dalam repositori ini untuk memudahkan navigasi:

halodoc-sentiment-analysis/
│
├── 📂 data/                          # Manajemen Data
│   ├── 📂 processed/                 # Data bersih & berlabel (CSV final)
│   └── 📂 raw/                       # Data mentah hasil scraping
│
├── 📂 model_halodoc_sentiment/       # (PENTING) Folder Artefak Model IndoBERT
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│   └── model.safetensors             # File bobot model (Wajib diunduh terpisah)
│
├── 📂 notebooks/                     # Eksperimen & Pelatihan Model
│   └── notebook.ipynb                # Kode lengkap training & evaluasi
│
├── app.py                            # File Utama Aplikasi Streamlit (Frontend UI)
├── inference.py                      # Logika Backend (Model Loader & Preprocessing)
├── requirements.txt                  # Daftar dependensi library Python
├── halodoc_reviews_labeled.csv       # Dataset default untuk dashboard
├── data_uji_deployment.xlsx          # Contoh data untuk pengujian upload
├── halodoc-logo-desktop.webp         # Aset gambar logo
└── README.md                         # Dokumentasi Proyek

## 🧠 Detail Model & Unduhan

Model yang digunakan adalah **IndoBERT Base P1** (`indobenchmark/indobert-base-p1`) yang telah melalui proses *Fine-Tuning*.

### Arsitektur & Pelatihan
*   **Base Model:** IndoBERT Base (12 layers, 768 hidden units).
*   **Metode Pelabelan:** *Pseudo-Labeling* menggunakan model *pre-trained* sentiment analysis untuk mengatasi bias rating bintang.
*   **Preprocessing:** Case folding, pembersihan *noise* (URL, user, angka), dan normalisasi kata tidak baku (*slang words*).
*   **Hyperparameters:** Epoch: 4, Batch Size: 32, Optimizer: AdamW, Learning Rate: 2e-5.

### 📥 Unduh Model (Wajib untuk Lokal)
Dikarenakan ukuran file model (`model.safetensors`) melebihi batas penyimpanan GitHub (>100MB), file tersebut disimpan di penyimpanan eksternal.

Jika Anda ingin menjalankan aplikasi ini di komputer lokal (Localhost), Anda **WAJIB** mengunduh file model melalui tautan di bawah ini:

👉 **[Download Model IndoBERT (Google Drive)](https://drive.google.com/file/d/1kZo_oIQb5CCycfGzpGoJEzCgZ84VFptl/view)**

**Instruksi Pemasangan Model:**
1.  Unduh file dari tautan di atas.
2.  Jika file berbentuk ZIP, ekstrak terlebih dahulu.
3.  Pastikan file `model.safetensors` (atau `pytorch_model.bin`) dimasukkan ke dalam folder `model_halodoc_sentiment/` di dalam direktori proyek ini.

## 🛠️ Instalasi & Penggunaan Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda:

1.  **Clone Repositori:**
    ```bash
    git clone https://github.com/LatiefDataVisionary/halodoc-sentiment-analysis.git
    cd halodoc-sentiment-analysis
    ```

2.  **Siapkan Environment (Opsional tapi Disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # atau
    venv\Scripts\activate      # Windows
    ```

3.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pastikan Model Terpasang:**
    Cek kembali apakah folder `model_halodoc_sentiment` sudah berisi file model yang diunduh dari bagian **Detail Model** di atas.

5.  **Jalankan Aplikasi:**
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

## 📊 Hasil Evaluasi

Model dievaluasi menggunakan *Test Set* terpisah (10% dari total data) dan menunjukkan performa yang sangat baik:

| Metrik | Nilai | Keterangan |
| :--- | :--- | :--- |
| **Akurasi** | **95.60%** | Kemampuan model memprediksi benar secara keseluruhan |
| **F1-Score (Positif)** | **98%** | Sangat baik dalam mengenali ulasan puas |
| **F1-Score (Negatif)** | **92%** | Sangat baik dalam mendeteksi keluhan |
| **F1-Score (Netral)** | **84%** | Cukup baik dalam mengenali ulasan ambigu |

## 📧 Kontak & Kontribusi

Proyek ini dikembangkan sebagai bagian dari Tugas Akhir Mata Kuliah **Teknik Pengembangan Model**. Jika Anda memiliki pertanyaan, saran, atau ingin berdiskusi, silakan hubungi:

*   **Lathif Ramadhan**
*   📩 Email: [datasciencelatief@gmail.com](mailto:datasciencelatief@gmail.com)
*   🐙 GitHub: [@LatiefDataVisionary](https://github.com/LatiefDataVisionary)

---
<p align="center">Made with ❤️ by Data Science Student</p>
