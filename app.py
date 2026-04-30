import streamlit as st
import requests
import io
from PIL import Image
import datetime

# --- KONFIGURATION & DESIGN ---
st.set_page_config(page_title="iFound | Premium Edition", layout="wide", initial_sidebar_state="collapsed")

# Dein Hugging Face Token (optional)
HEADERS = {"Authorization": "Bearer DEIN_TOKEN_HIER"} 
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

st.markdown("""
    <style>
    /* Hintergrund & Basis-Animation */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4); } 70% { box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); } }

    .stApp {
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), 
                    url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    /* SCHRIFT-OPTIMIERUNG: Weißer Text mit schwarzem Schatten für maximalen Kontrast */
    h1, h2, h3, p, label, .stMarkdown {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8) !important;
        animation: fadeIn 1s ease-out;
    }

    /* Apple-Style Home-Kacheln */
    div.stButton > button {
        height: 250px;
        width: 100% !important;
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(25px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 40px !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1) !important;
        animation: fadeIn 1.2s ease-out;
    }

    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: scale(1.05) translateY(-10px) !important;
        border: 1px solid rgba(255, 255, 255, 1) !important;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5) !important;
    }

    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Formular-Elemente Styling */
    .stTextInput input, .stFileUploader {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: white !important;
    }

    /* Navigation Back-Button */
    .back-btn button {
        height: auto !important;
        width: auto !important;
        padding: 12px 25px !important;
        font-size: 1.1rem !important;
        background: rgba(0,0,0,0.5) !important;
    }

    h1 { font-size: 6rem !important; letter-spacing: -3px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIK ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'inventar' not in st.session_state:
    st.session_state['inventar'] = []

def set_page(name):
    st.session_state['page'] = name

def query_ki(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
    return response.json()

# --- HOME ---
if st.session_state['page'] == 'home':
    st.markdown("<br><br><br><h1>iFound</h1><p style='font-size: 1.8rem;'>Entdecke das Verlorene.</p><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 2, 0.5])
    with col2:
        if st.button("🔎\nFund melden"):
            set_page('melden')
            st.rerun()
    with col4:
        if st.button("📂\nArchiv öffnen"):
            set_page('archiv')
            st.rerun()

# --- MELDEN ---
elif st.session_state['page'] == 'melden':
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Zurück"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2>Gegenstand einreichen</h2>", unsafe_allow_html=True)
    
    col_up, col_pre = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
        ort = st.text_input("Wo wurde es gefunden?")

    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes))
        with col_pre:
            st.image(image, width=300)
            if st.button("✨ KI-Scan starten"):
                with st.spinner("Analysiere Moleküle..."):
                    try:
                        res = query_ki(img_bytes)
                        label = res[0]['label'].split(",")[0].capitalize()
                        st.session_state['temp_item'] = label
                        st.success(f"Objekt erkannt: {label}")
                    except:
                        st.error("KI schläft gerade.")
            
            if 'temp_item' in st.session_state:
                if st.button("✅ Sicher aufbewahren"):
                    st.session_state['inventar'].append({
                        "name": st.session_state['temp_item'],
                        "ort": ort,
                        "bild": image,
                        "zeit": datetime.datetime.now().strftime("%H:%M - %d.%m.%y")
                    })
                    st.balloons()
                    del st.session_state['temp_item']
                    set_page('home')
                    st.rerun()

# --- ARCHIV ---
elif st.session_state['page'] == 'archiv':
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Home"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2>Alle Fundstücke</h2>", unsafe_allow_html=True)
    if not st.session_state['inventar']:
        st.info("Das Archiv ist leer.")
    else:
        for item in reversed(st.session_state['inventar']):
            with st.expander(f"📦 {item['name']} | {item['ort']}"):
                c1, c2 = st.columns([1, 2])
                c1.image(item['bild'], width=150)
                c2.write(f"**Registriert am:** {item['zeit']}")
