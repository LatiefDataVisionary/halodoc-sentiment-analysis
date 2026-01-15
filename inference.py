import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import os
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import io

# ==========================================
# 1. KONFIGURASI & RESOURCE
# ==========================================
# Mendapatkan lokasi file inference.py saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Menggabungkan lokasi file dengan nama folder model
MODEL_PATH = os.path.join(BASE_DIR, "model_halodoc_sentiment")
DATA_PATH = os.path.join(BASE_DIR, "halodoc_reviews_labeled.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOADED = False

# Kamus Normalisasi (Slang) - Lengkap
NORMALISASI_KAMUS = {
    'yg': 'yang', 'gk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak',
    'bgt': 'banget', 'dr': 'dokter', 'kalo': 'kalau', 'klo': 'kalau',
    'blm': 'belum', 'sdh': 'sudah', 'udh': 'sudah', 'dgn': 'dengan',
    'tdk': 'tidak', 'tpi': 'tapi', 'tp': 'tapi', 'krn': 'karena',
    'pke': 'pakai', 'pake': 'pakai', 'sy': 'saya', 'aku': 'saya',
    'gw': 'saya', 'gue': 'saya', 'jd': 'jadi', 'jdi': 'jadi',
    'bisa': 'bisa', 'bs': 'bisa', 'dpt': 'dapat', 'jgn': 'jangan',
    'utk': 'untuk', 'nya': 'nya', 'bnyk': 'banyak', 'dlm': 'dalam',
    'bgus': 'bagus', 'keren': 'keren', 'mantap': 'mantap',
    'min': 'admin', 'apk': 'aplikasi', 'app': 'aplikasi',
    'error': 'eror', 'lemot': 'lambat', 'lelet': 'lambat',
    'konsul': 'konsultasi', 'obat': 'obat', 'resep': 'resep',
    'chat': 'pesan', 'bales': 'balas', 'respon': 'respons',
    'cepet': 'cepat', 'cpt': 'cepat', 'mksih': 'terima kasih',
    'makasih': 'terima kasih', 'tks': 'terima kasih', 'thx': 'terima kasih',
    'good': 'bagus', 'bad': 'buruk', 'best': 'terbaik',
    'oke': 'oke', 'ok': 'oke', 'sip': 'sip',
    'mw': 'mau', 'mau': 'mau', 'trus': 'terus', 'trs': 'terus',
    'lg': 'lagi', 'lgi': 'lagi', 'sm': 'sama', 'sama': 'sama',
    'd': 'di', 'k': 'ke', 'org': 'orang', 'mlm': 'malam',
    'pagi': 'pagi', 'siang': 'siang', 'sore': 'sore'
}

# ==========================================
# 2. PREPROCESSING & MODEL LOADER
# ==========================================
def normalisasi_kata(text):
    """Mengubah kata tidak baku menjadi baku berdasarkan kamus."""
    words = text.split()
    normalized_words = [NORMALISASI_KAMUS.get(word, word) for word in words]
    return ' '.join(normalized_words)

def preprocess_text(text):
    """Membersihkan teks dari noise (URL, Angka, Simbol, Slang)."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Hapus URL
    text = re.sub(r'@\w+|#\w+', '', text) # Hapus Mention/Hashtag
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Hapus angka & simbol
    text = normalisasi_kata(text) # Normalisasi Slang
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    
    return text

@st.cache_resource
def load_model():
    """Memuat Model IndoBERT & Tokenizer (Cached)."""
    global MODEL_LOADED
    print(f"🔄 Mencoba memuat model dari: {MODEL_PATH}") # Debugging Log
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Folder '{MODEL_PATH}' tidak ditemukan!")
            
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        
        # PENTING: use_safetensors=True agar bisa baca file .safetensors
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH, 
            use_safetensors=True
        )
        
        model.to(DEVICE)
        model.eval()
        
        MODEL_LOADED = True
        print("✅ Model berhasil dimuat!")
        return tokenizer, model
    except Exception as e:
        MODEL_LOADED = False
        print(f"❌ Error loading model: {e}")
        return None, None

@st.cache_data
def load_data():
    """Memuat data CSV default untuk dashboard."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            return df
        else:
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load model saat file diimport
tokenizer, model = load_model()

# ==========================================
# 3. ENGINE PREDIKSI (SINGLE & BATCH)
# ==========================================
def predict_sentiment(text):
    """
    Prediksi untuk satu kalimat (Live Prediction).
    Output: Label, Confidence Score, List Probabilitas
    """
    if not MODEL_LOADED:
        return "Error", 0.0, [0, 0, 0]
    
    clean_text = preprocess_text(text)
    
    inputs = tokenizer(
        clean_text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        
    conf, pred = torch.max(probs, dim=1)
    
    label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    return label_map[pred.item()], conf.item(), probs.cpu().numpy()[0]

@st.cache_data(show_spinner=False)
def predict_batch(df, text_column):
    """
    Prediksi massal untuk DataFrame (File Upload).
    Menggunakan Batch Processing agar hemat memori & cepat.
    """
    if not MODEL_LOADED:
        return df
    
    # 1. Preprocessing Massal
    # Kita buat kolom baru 'clean_text'
    df['clean_text'] = df[text_column].astype(str).apply(preprocess_text)
    
    # Hapus data kosong setelah cleaning
    df = df[df['clean_text'].str.strip() != ""]
    
    labels = []
    confidences = []
    
    # 2. Batch Inference
    batch_size = 32 # Sesuaikan dengan VRAM GPU/RAM CPU
    texts = df['clean_text'].tolist()
    
    # Progress bar placeholder (opsional, bisa dihandle di app.py)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            max_length=128, 
            padding=True, 
            truncation=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
        label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        
        labels.extend([label_map[p.item()] for p in pred])
        confidences.extend([c.item() for c in conf])
        
    # 3. Simpan Hasil
    df['sentiment_pred'] = labels
    df['confidence_score'] = confidences
    
    return df

# ==========================================
# 4. ENGINE VISUALISASI (WORDCLOUD & N-GRAM)
# ==========================================
def generate_wordcloud(text_data, title, colormap='viridis'):
    """
    Membuat WordCloud Image.
    Mengembalikan objek Figure Matplotlib.
    """
    if len(text_data) == 0:
        return None
        
    text = " ".join(text_data.astype(str))
    
    # Konfigurasi WordCloud Transparan & Keren
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color=None, # Transparan
        mode="RGBA",
        colormap=colormap,
        max_words=150,
        contour_width=0,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    # Set background figure jadi transparan agar menyatu dengan dark mode
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    return fig

def get_top_bigrams(text_data, n=10):
    """
    Menghitung frekuensi 2 kata (Bigram) terbanyak.
    Mengembalikan DataFrame untuk divisualisasikan dengan Altair.
    """
    if len(text_data) < 2:
        return pd.DataFrame(columns=['Bigram', 'Frekuensi'])
    
    # Menggunakan CountVectorizer dari Scikit-Learn
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=None).fit(text_data)
    bag_of_words = vec.transform(text_data)
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    # Ambil Top N
    df_bigram = pd.DataFrame(words_freq[:n], columns=['Bigram', 'Frekuensi'])
    return df_bigram

# ==========================================
# 5. UTILITIES (DOWNLOAD)
# ==========================================
def convert_df_to_csv(df):
    """Mengubah DataFrame ke CSV bytes untuk tombol download."""
    return df.to_csv(index=False).encode('utf-8')