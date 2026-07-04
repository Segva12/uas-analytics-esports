import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit.components.v1 as components
import os

# Set konfigurasi halaman utama dashboard
st.set_page_config(page_title="E-Sports Toxicity & Network Analytics", page_icon="🎮", layout="wide")

# =================================================================
# LOAD DATA & MODEL (Menggunakan Cache agar Dashboard Cepat)
# =================================================================
@st.cache_data
def load_data():
    df_main = pd.read_csv("dataset_dashboard.csv")
    df_central = pd.read_csv("sna_centrality_results.csv")
    return df_main, df_central

@st.cache_resource
def load_models():
    tfidf = joblib.load("vectorizer_tfidf.pkl")
    nb = joblib.load("model_naive_bayes.pkl")
    svm = joblib.load("model_svm.pkl")
    rf = joblib.load("model_random_forest.pkl")
    return tfidf, nb, svm, rf

# Proteksi error jika file belum diunduh lengkap
try:
    df, df_centrality = load_data()
    vectorizer, nb_model, svm_model, rf_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat komponen dashboard. Pastikan semua file .csv, .pkl, .html, dan .png hasil export dari Colab sudah berada di folder yang sama dengan app.py. Error: {e}")
    st.stop()

# =================================================================
# SIDEBAR NAVIGATION
# =================================================================
st.sidebar.title("📌 Menu Navigasi")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    [
        "📊 Overview & Distribusi Data", 
        "🤖 Komparasi Model & Pengujian Live",
        "🎯 Klasterisasi Isu (BERTopic)",
        "🕸️ Jaringan Komunitas (SNA)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("🚀 **Project UAS - Analisis Sentimen Komunitas E-Sports**")

# =================================================================
# HALAMAN 1: OVERVIEW & DISTRIBUSI DATA
# =================================================================
if menu == "📊 Overview & Distribusi Data":
    st.title("🎮 E-Sports Toxicity Analytics Dashboard")
    st.subheader("Analisis Ujaran Kebencian dan Sentimen Komunitas Gamer")
    st.markdown("---")

    total_komentar = len(df)
    toxic_count = int(df['Label'].value_counts().get(1, 0))
    nontoxic_count = int(df['Label'].value_counts().get(0, 0))
    toxic_percentage = (toxic_count / total_komentar) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Data Komentar", value=f"{total_komentar} baris")
    with col2:
        st.metric(label="Komentar Toxic", value=toxic_count, delta="Ujaran Kebencian", delta_color="inverse")
    with col3:
        st.metric(label="Komentar Aman (Non-Toxic)", value=nontoxic_count)
    with col4:
        st.metric(label="Rasio Toksisitas", value=f"{toxic_percentage:.2f}%")

    st.markdown("---")

    # Tata Letak Baru: 2 Kolom untuk Bar Chart dan Pie Chart
    col_bar, col_pie = st.columns(2)
    
    with col_bar:
        st.markdown("#### 📈 Bar Chart Distribusi Label")
        label_counts = df['Label'].value_counts().rename(index={0: 'Non-Toxic', 1: 'Toxic'})
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=label_counts.index, y=label_counts.values, palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xlabel('Kategori Komentar', fontsize=10)
        ax.set_ylabel('Jumlah Data', fontsize=10)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
                        xytext=(0, 3), textcoords='offset points')
        st.pyplot(fig)

    with col_pie:
        st.markdown("#### 🥧 Pie Chart Persentase Toksisitas")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        labels = ['😇 Non-Toxic', '🤬 Toxic']
        sizes = [nontoxic_count, toxic_count]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.08)  # Efek potongan terpisah untuk menonjolkan bagian Toxic
        
        ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=140, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax_pie.axis('equal')  # Memastikan pie chart berbentuk lingkaran sempurna
        st.pyplot(fig_pie)

    st.markdown("---")
    
    # Tabel diubah menjadi lebar penuh di bawah grafik
    st.markdown("#### 🔍 Eksplorasi Database Komentar")
    search_query = st.text_input("Cari kata kunci spesifik dalam tabel komentar (Contoh: onic, btr, hater):")
    
    df_display = df[['Username', 'Clean_Teks', 'Label']].copy()
    df_display['Label'] = df_display['Label'].map({0: '😇 Non-Toxic', 1: '🤬 Toxic'})
    
    if search_query:
        df_display = df_display[df_display['Clean_Teks'].str.contains(search_query, case=False, na=False)]
    
    st.dataframe(df_display, use_container_width=True, height=280)

# =================================================================
# HALAMAN 2: KOMPARASI MODEL & PENGUJIAN LIVE
# =================================================================
elif menu == "🤖 Komparasi Model & Pengujian Live":
    st.title("🤖 Evaluasi Kecerdasan Buatan & Klasifikasi Teks")
    st.markdown("---")

    st.markdown("### 🧪 Simulator Deteksi Toksisitas Real-Time")
    st.write("Ketik komentar e-sports apa saja di bawah ini. Model AI terbaik (**SVM**) akan langsung memprediksi kategori teks secara otomatis:")
    
    input_teks = st.text_input("Input Kalimat Pengujian:", placeholder="Ketik di sini... (Contoh: tim rrq cacat ganiat main atau GG onic)")
    
    if input_teks:
        teks_transformed = vectorizer.transform([input_teks])
        try:
            decision_score = svm_model.decision_function(teks_transformed)[0]
            prediksi_live = 1 if decision_score > 0 else 0
        except:
            prediksi_live = rf_model.predict(teks_transformed)[0]
        
        st.markdown("#### Hasil Analisis Model AI:")
        if prediksi_live == 1:
            st.error(f"❌ **Komentar Terdeteksi: TOXIC (Ujaran Kebencian)**\n\nKalimat: \"{input_teks}\"")
        else:
            st.success(f"✅ **Komentar Terdeteksi: NON-TOXIC (Aman/Bersih)**\n\nKalimat: \"{input_teks}\"")
            
    st.markdown("---")

    st.markdown("### 📊 Papan Skor Kompetisi Algoritma (Metrics Evaluation)")
    
    # Karena tabel sekarang lebih lebar, kita buat col_metrics sedikit lebih luas
    col_metrics, col_matrix = st.columns([1.3, 1])
    
    with col_metrics:
        st.write("Perbandingan performa menyeluruh dari 3 model klasifikasi berdasarkan data pengujian:")
        
        # Penambahan Precision, Recall, dan F1-Score berdasarkan Weighted Avg dari output Colab-mu
        df_compare = pd.DataFrame({
            'Algoritma Model': ['Support Vector Machine (SVM)', 'Random Forest Classifier', 'Multinomial Naive Bayes'],
            'Accuracy': ['80.80%', '75.45%', '67.41%'],
            'Precision': ['81.00%', '76.00%', '70.00%'],
            'Recall': ['81.00%', '75.00%', '67.00%'],
            'F1-Score': ['81.00%', '76.00%', '64.00%']
        })
        
        # Menggunakan st.dataframe agar tabel terlihat rapi dan tidak terpotong
        st.dataframe(df_compare, use_container_width=True, hide_index=True)
        
        st.success("💡 **Rekomendasi Utama:** Algoritma **Support Vector Machine (SVM)** terpilih sebagai model utama karena unggul di segala metrik evaluasi. F1-Score yang mencapai 81% membuktikan SVM sangat seimbang dalam menekan *False Positive* maupun *False Negative*.")

    with col_matrix:
        st.write("#### Struktur Kebenaran Tebakan (Confusion Matrix)")
        pilihan_matrix = st.selectbox("Pilih Confusion Matrix Model yang Ingin Ditampilkan:", ["SVM (Pemenang)", "Random Forest", "Naive Bayes"])
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
        if pilihan_matrix == "Naive Bayes":
            matrix_data = [[34, 62], [11, 117]]
            cmap_color = 'Blues'
            title_cm = 'Confusion Matrix - Naive Bayes'
        elif pilihan_matrix == "Random Forest":
            matrix_data = [[74, 22], [33, 95]]
            cmap_color = 'Greens'
            title_cm = 'Confusion Matrix - Random Forest'
        else:
            matrix_data = [[79, 17], [26, 102]]
            cmap_color = 'Oranges'
            title_cm = 'Confusion Matrix - SVM'
            
        sns.heatmap(matrix_data, annot=True, fmt='d', cmap=cmap_color, ax=ax_cm,
                    xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
        ax_cm.set_title(title_cm, fontsize=11, fontweight='bold')
        ax_cm.set_ylabel('Aktual (Label Asli)')
        ax_cm.set_xlabel('Prediksi (Tebakan Mesin)')
        st.pyplot(fig_cm)

# =================================================================
# HALAMAN 3: KLASTERISASI ISU (BERTopic)
# =================================================================
elif menu == "🎯 Klasterisasi Isu (BERTopic)":
    st.title("🎯 Topic Modeling: Pemetaan Klaster Isu E-Sports")
    st.markdown("**Metode:** BERTopic (Multilingual) + UMAP (random_state=42) + HDBSCAN")
    st.markdown("---")

    # Bagian Barchart dan Penjelasan Tabel Topik
    st.markdown("### 📊 Kata Kunci Utama per Klaster Isu (Filtered)")
    st.write("Topik Berkualitas yang Ditemukan (dari 9 total topik):")
    
    # Membuat tabel rincian topik sesuai gambar presentasi
    df_topik = pd.DataFrame({
        "Topic ID": [0, 4, 5, 6],
        "Nama Topik": [
            "Diskusi Fans ONIC vs BTR", 
            "Ejekan tim rrq ke tim btr", 
            "Gameplay Pemain Import PH", 
            "Meme Hertod"
        ],
        "Kata Kunci Utama": [
            "onic, btr, fans, juara, kalah", 
            "rrq, kelinci, transfer, pemain", 
            "import, ph, gameplay, indonesia", 
            "hertod, coach, timnas, meme"
        ]
    })
    
    # Menampilkan tabel tanpa index agar lebih rapi
    st.dataframe(df_topik, use_container_width=True, hide_index=True)

    try:
        with open("barchart_topik.html", 'r', encoding='utf-8') as f_bar:
            html_bar = f_bar.read()
        components.html(html_bar, height=380, scrolling=False)
    except Exception as e:
        st.error(f"Berkas barchart_topik.html tidak ditemukan. Error: {e}")

    st.markdown("---")

    # Bagian Tren Line Chart dan Penjelasan Temuan
    st.markdown("### 📈 Deteksi Ledakan Isu E-Sports (Burst) per Hari")
    st.write("Visualisasi: Line chart tren topik per hari (10–23 Juni 2026)")
    
    try:
        with open("trend_kalender_topik.html", 'r', encoding='utf-8') as f_trend:
            html_trend = f_trend.read()
        components.html(html_trend, height=520, scrolling=False)
    except Exception as e:
        st.error(f"Berkas trend_kalender_topik.html tidak ditemukan. Error: {e}")

    # Menambahkan kotak informasi temuan analisis tren
    st.info("""
    **💡 Temuan Analisis:**
    * Topik **ONIC vs BTR** paling dominan dan konsisten sepanjang periode.
    * Topik **Pemain Import PH** relatif stabil.
    * Topik **Meme Hertod** muncul sporadis mengikuti momen viral tertentu.
    * Topik **RRQ/Kelinci** relatif stabil dengan volume sedang.
    """)

# =================================================================
# HALAMAN 4: SOCIAL NETWORK ANALYSIS (SNA) - UPDATED WITH POLARIZATION
# =================================================================
elif menu == "🕸️ Jaringan Komunitas (SNA)":
    st.title("🕸️ Social Network Analysis: Jaringan Komunitas & Polarisasi Fandom")
    st.write("Analisis struktural jaringan untuk memetakan peran aktor kunci (Centrality) serta mengukur tingkat gesekan atau polarisasi antar fanbase tim besar.")
    st.markdown("---")

    # Membuat Tab agar Dashboard rapi dan tidak memanjang ke bawah
    tab_polarisasi, tab_louvain = st.tabs(["🔥 1. Analisis Polarisasi Fanbase", "🧩 2. Komunitas Jaringan (Louvain)"])

    # ==========================================
    # TAB 1: ANALISIS POLARISASI (YANG BARU)
    # ==========================================
    with tab_polarisasi:
        st.markdown("### 📊 Peta Polarisasi dan Gesekan Antar Kubu Isu Tim")
        st.write("Grafik ini menonjolkan bagaimana netizen bergerak di sekitar isu tim-tim besar (ONIC, BTR, RRQ) untuk mendeteksi seberapa jauh jarak perpecahan opini mereka:")
        
        col_img_pol, col_info_pol = st.columns([1.3, 1])
        
        with col_img_pol:
            if os.path.exists("sna_polarisasi_fandom.png"):
                st.image("sna_polarisasi_fandom.png", use_container_width=True, 
                         caption="Graf Polarisasi Jaringan Netizen terhadap Isu Tim Besar")
            else:
                st.warning("⚠️ Berkas 'sna_polarisasi_fandom.png' tidak ditemukan. Pastikan sudah dirun plt.savefig() di Colab dan filenya sudah didownload ke folder app.py.")
        
        with col_info_pol:
            st.markdown("#### 🏆 Pengaruh Aktor Utama Jaringan (Centrality)")
            pilihan_tabel = st.selectbox("Pilih Analisis Perilaku Akun Netizen:", 
                                         ["Top KOL (Betweenness Centrality)", "Top Penyebar Toxic (Toxic Count)"],
                                         key="sb_centrality")
            
            df_show = df_centrality[['Username', 'Degree_Centrality', 'Betweenness_Centrality', 'Toxic_Count', 'Dominasi']].copy()
            
            if pilihan_tabel == "Top KOL (Betweenness Centrality)":
                st.write("Akun yang menjadi **'Jembatan Informasi'** lintas kubu/topik pembicaraan (Skor Betweenness tertinggi):")
                df_show = df_show.sort_values(by='Betweenness_Centrality', ascending=False).head(5)
            else:
                st.write("Akun yang paling banyak memproduksi komentar dengan **Label Toxic** di database:")
                df_show = df_show.sort_values(by='Toxic_Count', ascending=False).head(5)
                
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            
            st.success("""
            * **User di Pinggiran Node Isu:** Merupakan indikator **Polarisasi Fanatik**, di mana akun netizen tersebut hanya berinteraksi di dalam satu ekosistem tim saja dan enggan menyeberang ke isu tim lain.
            * **User di Tengah-Tengah Antar Node Isu:** Merupakan aktor penggerak opini lintas kubu. Di titik pertemuan inilah gesekan sentimen atau perdebatan panas antar-fandom paling rawan terjadi.
            """)

    # ==========================================
    # TAB 2: COMMUNITY DETECTION (LOUVAIN)
    # ==========================================
    with tab_louvain:
        st.markdown("### 🧮 Peta Pengelompokan Sub-Isu Menggunakan Algoritma Louvain")
        st.write("Algoritma Louvain memecah struktur jaringan menjadi klaster modularitas yang lebih spesifik berdasarkan kedekatan interaksi:")
        
        col_img_lou, col_info_lou = st.columns([1.3, 1])
        
        with col_img_lou:
            if os.path.exists("sna_community_detection.png"):
                st.image("sna_community_detection.png", use_container_width=True,
                         caption="Visualisasi Deteksi Komunitas Louvain (Tema Gelap)")
            else:
                st.warning("⚠️ Berkas 'sna_community_detection.png' tidak ditemukan di folder lokal dashboard kamu.")
                
            st.info("📊 **Metrik Global Jaringan (SNA):**\n\n* **Total Nodes (User + Topik):** 70\n* **Total Edges (Koneksi):** 77\n* **Density (Kerapatan Jaringan):** 0.0319\n* **Modularity Score:** 0.2331")

        with col_info_lou:
            st.markdown("""
            📊 **Komposisi Isu Komunitas Dominan:**
            * **Komunitas 1 (Kubu Utama - 42 User):** Berpusat penuh pada topik *Diskusi Fans ONIC vs BTR*. Klaster ini memiliki **Toxic Rate sebesar 66.3%**.
            * **Komunitas 6 (Kubu Konflik - 5 User):** Berpusat pada topik *Hujatan Fandom RRQ ke Tim BTR*. Kelompok ini mencatat **Toxic Rate tertinggi sebesar 81.8%**, membuktikan area ini sebagai titik konflik terpanas di komunitas.
            * **Komunitas 0 (Kubu Analis - 1 User):** Berpusat pada topik *Pemain Import PH dan Perbandingan Makro Gameplay*.
            """)
