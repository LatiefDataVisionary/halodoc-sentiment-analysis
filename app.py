import streamlit as st
import pandas as pd
import altair as alt
import time
import matplotlib.pyplot as plt

# Import modul logika (Pastikan file inference.py ada di folder yang sama)
import inference

# ==========================================
# 1. KONFIGURASI HALAMAN (PAGE CONFIG)
# ==========================================
st.set_page_config(
    page_title="Halodoc Sentiment AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS MODERN & GLASSMORPHISM
# ==========================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Background Utama Gelap Elegan */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Cards (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #E0004D;
    }

    /* Typography */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800;
    }
    p, label, .stMarkdown {
        color: #e0e0e0 !important;
    }

    /* Metric Styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ff4b8b, #E0004D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #a0a0b0;
        margin-bottom: 5px;
    }

    /* Custom Button */
    .stButton > button {
        background: linear-gradient(90deg, #E0004D 0%, #ff4b8b 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(224, 0, 77, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(224, 0, 77, 0.6);
    }
    
    /* File Uploader */
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI UI REUSABLE (DASHBOARD COMPONENT)
# ==========================================
def render_dashboard_ui(df):
    """
    Fungsi ini merender seluruh komponen dashboard (KPI, Grafik, WordCloud).
    Bisa dipakai untuk data default maupun data hasil upload.
    """
    # 1. Normalisasi Kolom Label
    # Cek apakah kolom label bernama 'label' (data lama) atau 'sentiment_pred' (data baru)
    label_col = 'label' if 'label' in df.columns else 'sentiment_pred'
    
    # Cek apakah kolom teks bernama 'content' atau 'clean_text' (hasil preprocessing)
    text_col = 'content' if 'content' in df.columns else 'clean_text'
    if text_col not in df.columns and 'Review Text' in df.columns: text_col = 'Review Text'

    # Mapping label angka ke string jika masih berupa angka (0,1,2)
    if df[label_col].dtype != 'O': 
         label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
         df['label_str'] = df[label_col].map(label_map)
         label_col = 'label_str'

    # 2. Hitung Metrik KPI
    total_reviews = len(df)
    pos_reviews = len(df[df[label_col] == 'Positif'])
    neu_reviews = len(df[df[label_col] == 'Netral'])
    neg_reviews = len(df[df[label_col] == 'Negatif'])

    # 3. Tampilkan Kartu KPI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Total Ulasan</div><div class="metric-value">{total_reviews:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Positif 😊</div><div class="metric-value" style="color: #4CAF50;">{pos_reviews:,}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Netral 😐</div><div class="metric-value" style="color: #FFC107;">{neu_reviews:,}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="glass-card"><div class="metric-label">Negatif 😡</div><div class="metric-value" style="color: #F44336;">{neg_reviews:,}</div></div>""", unsafe_allow_html=True)

    # 4. Tab Visualisasi
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Sentimen", "☁️ Word Cloud", "🔠 Analisis N-Gram"])
    
    # --- Tab 1: Donut Chart ---
    with tab1:
        col_chart1, col_chart2 = st.columns([2, 1])
        with col_chart1:
            st.markdown("### Proporsi Sentimen")
            source = pd.DataFrame({
                'Sentimen': ['Negatif', 'Netral', 'Positif'],
                'Jumlah': [neg_reviews, neu_reviews, pos_reviews],
                'Color': ['#F44336', '#FFC107', '#4CAF50']
            })
            
            base = alt.Chart(source).encode(theta=alt.Theta("Jumlah", stack=True))
            pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                color=alt.Color("Sentimen", scale=alt.Scale(domain=['Negatif', 'Netral', 'Positif'], range=['#F44336', '#FFC107', '#4CAF50'])),
                order=alt.Order("Jumlah", sort="descending"),
                tooltip=["Sentimen", "Jumlah"]
            )
            text = base.mark_text(radius=140).encode(
                text="Jumlah", order=alt.Order("Jumlah", sort="descending"), color=alt.value("white")
            )
            st.altair_chart((pie + text).properties(height=400), use_container_width=True)
        
        with col_chart2:
            st.markdown("### Sampel Data")
            # Tampilkan tabel preview
            st.dataframe(df[[text_col, label_col]].head(10), hide_index=True, use_container_width=True)

    # --- Tab 2: Word Cloud ---
    with tab2:
        st.markdown("### Visualisasi Kata Kunci (Word Cloud)")
        sentiment_filter = st.selectbox("Pilih Sentimen:", ["Positif", "Negatif", "Netral"], key="wc_select")
        
        # Filter data berdasarkan sentimen
        text_data = df[df[label_col] == sentiment_filter][text_col]
        
        if len(text_data) > 0:
            with st.spinner("Membuat Word Cloud..."):
                fig = inference.generate_wordcloud(text_data, f"Word Cloud - {sentiment_filter}")
                st.pyplot(fig)
        else:
            st.info("Tidak ada data untuk sentimen ini.")

    # --- Tab 3: N-Gram ---
    with tab3:
        st.markdown("### Frasa Paling Sering Muncul (Bigram)")
        sentiment_filter_ngram = st.selectbox("Pilih Sentimen untuk N-Gram:", ["Negatif", "Positif"], key="ngram_select")
        
        text_data_ngram = df[df[label_col] == sentiment_filter_ngram][text_col]
        
        if len(text_data_ngram) > 0:
            with st.spinner("Menganalisis N-Gram..."):
                # Pastikan data teks bersih (string)
                text_data_ngram = text_data_ngram.astype(str)
                df_bigram = inference.get_top_bigrams(text_data_ngram, n=10)
                
                if not df_bigram.empty:
                    chart = alt.Chart(df_bigram).mark_bar().encode(
                        x='Frekuensi',
                        y=alt.Y('Bigram', sort='-x'),
                        color=alt.value('#E0004D'),
                        tooltip=['Bigram', 'Frekuensi']
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Data tidak cukup untuk membentuk Bigram.")
        else:
            st.info("Data tidak cukup untuk analisis N-Gram.")

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # Logo (Ganti dengan path logo lokal jika ada, atau URL)
    st.image("halodoc-logo-desktop.webp", width=180) 
    st.markdown("### AI HALODOC Sentiment Engine")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigasi",
        ["Dashboard Analisis", "Analisis File (Batch)", "Uji Coba Model (Live)", "Tentang Model"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**System Status**")
    
    model_status = st.empty()
    if inference.MODEL_LOADED:
        model_status.success(f"🟢 ONLINE ({str(inference.DEVICE).upper()})")
    else:
        model_status.error("🔴 OFFLINE (Model Error)")
        
    st.caption("© 2026 Project UAS Mata Kuliah Teknik Pengembagan Model Prodi Sains Data")

# ==========================================
# 5. HALAMAN: DASHBOARD ANALISIS (DEFAULT)
# ==========================================
if selected_page == "Dashboard Analisis":
    st.title("📊 Dashboard Analisis Sentimen")
    st.markdown("Monitoring performa ulasan aplikasi Halodoc secara real-time berdasarkan data historis.")

    # Load Data CSV Default
    df = inference.load_data()

    if df is not None:
        # PANGGIL FUNGSI UI REUSABLE
        render_dashboard_ui(df)
    else:
        st.warning("⚠️ Data CSV default tidak ditemukan. Silakan upload file di menu 'Analisis File'.")

# ==========================================
# 6. HALAMAN: ANALISIS FILE (BATCH)
# ==========================================
elif selected_page == "Analisis File (Batch)":
    st.title("📂 Analisis File Massal")
    st.markdown("Upload file CSV atau Excel berisi ulasan, AI akan menganalisisnya, dan **Dashboard Lengkap** akan muncul.")
    
    uploaded_file = st.file_uploader("Upload File (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            st.success(f"File berhasil diupload! Total baris: {len(df_upload)}")
            st.dataframe(df_upload.head())
            
            text_col = st.selectbox("Pilih Kolom yang berisi Ulasan:", df_upload.columns)
            
            if st.button("🚀 Mulai Analisis Batch"):
                if not inference.MODEL_LOADED:
                    st.error("Model belum siap.")
                else:
                    with st.spinner("Sedang memproses... (Ini mungkin memakan waktu tergantung jumlah data)"):
                        start_time = time.time()
                        
                        # Proses Prediksi
                        df_result = inference.predict_batch(df_upload, text_col)
                        
                        end_time = time.time()
                        duration = end_time - start_time
                        
                    st.success(f"✅ Analisis Selesai dalam {duration:.2f} detik!")
                    
                    # --- FITUR BARU: TAMPILKAN DASHBOARD LENGKAP ---
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Analisis File Anda")
                    
                    # Panggil fungsi UI yang sama dengan Dashboard Utama!
                    render_dashboard_ui(df_result)
                    
                    # --- DOWNLOAD SECTION ---
                    st.markdown("---")
                    st.markdown("### 📥 Download Data Hasil")
                    csv_data = inference.convert_df_to_csv(df_result)
                    st.download_button(
                        label="⬇️ Download Hasil Analisis (CSV)",
                        data=csv_data,
                        file_name='hasil_analisis_sentimen_halodoc_batch.csv',
                        mime='text/csv',
                    )
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ==========================================
# 7. HALAMAN: UJI COBA MODEL (LIVE)
# ==========================================
elif selected_page == "Uji Coba Model (Live)":
    st.title("🧪 Uji Coba Model (Live Prediction)")
    st.markdown("Ketik ulasan di bawah ini untuk melihat bagaimana **IndoBERT** menganalisis sentimennya.")

    col_input, col_result = st.columns([1.5, 1])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        user_input = st.text_area("Masukkan Ulasan Pengguna:", height=150, placeholder="Contoh: Aplikasinya bagus banget, dokter ramah tapi obatnya agak mahal...")
        
        analyze_btn = st.button("🔍 Analisis Sentimen")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if analyze_btn and user_input:
            if not inference.MODEL_LOADED:
                st.error("Model belum siap. Cek log error.")
            else:
                with st.spinner('Sedang memproses dengan IndoBERT...'):
                    time.sleep(0.5)
                    label, confidence, probs = inference.predict_sentiment(user_input)
                
                color_map = {'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'}
                result_color = color_map.get(label, '#ffffff')

                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-top: 5px solid {result_color};">
                    <h3 style="margin-bottom: 0;">Hasil Prediksi</h3>
                    <h1 style="font-size: 3.5rem; color: {result_color} !important; margin: 10px 0;">{label}</h1>
                    <p>Confidence Score: <b>{confidence:.2%}</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Detail Probabilitas")
                st.caption("Negatif")
                st.progress(float(probs[0]))
                st.caption("Netral")
                st.progress(float(probs[1]))
                st.caption("Positif")
                st.progress(float(probs[2]))

        elif analyze_btn and not user_input:
            st.warning("⚠️ Mohon ketik ulasan terlebih dahulu.")
            
        else:
            st.info("👈 Silakan masukkan teks di sebelah kiri untuk memulai.")

# ==========================================
# 8. HALAMAN: TENTANG MODEL
# ==========================================
elif selected_page == "Tentang Model":
    st.title("🤖 Tentang Model IndoBERT")
    
    st.markdown("""
    <div class="glass-card">
        <h3>Arsitektur Model</h3>
        <p>Model ini dibangun menggunakan arsitektur <b>IndoBERT Base P1</b> yang dikembangkan oleh IndoBenchmark. 
        Model ini telah melalui proses <i>Fine-Tuning</i> menggunakan dataset ulasan Halodoc sebanyak 50.000+ data.</p>
        <ul>
            <li><b>Base Model:</b> indobenchmark/indobert-base-p1</li>
            <li><b>Epochs:</b> 4</li>
            <li><b>Accuracy:</b> 95.6%</li>
            <li><b>Optimizer:</b> AdamW</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
    st.caption("Powered by Hugging Face Transformers & PyTorch")