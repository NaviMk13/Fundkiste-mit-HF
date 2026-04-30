import streamlit as st
import requests
import io
from PIL import Image
import datetime
import time

# --- KONFIGURATION & DESIGN ---
st.set_page_config(page_title="iFound | Premium Edition", layout="wide", initial_sidebar_state="collapsed")

# Stabileres Modell für anonyme Anfragen
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"

st.markdown("""
    <style>
    /* Hintergrund & Animationen */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                    url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    /* TEXT-LESBARKEIT: Starker Schatten für den Wald-Hintergrund */
    h1, h2, h3, p, label, .stMarkdown {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 10px rgba(0,0,0,1), 0px 0px 20px rgba(0,0,0,0.5) !important;
        animation: fadeIn 0.8s ease-out;
    }

    /* RIESIGE HOME-KACHELN (Apple Style) */
    div.stButton > button {
        height: 280px;
        width: 100% !important;
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 40px !important;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }

    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: scale(1.05) translateY(-10px) !important;
        border: 1px solid rgba(255, 255, 255, 1) !important;
        box-shadow: 0 30px 60px rgba(0,0,0,0.6) !important;
    }

    /* Styling für Upload & Input */
    .stTextInput input, .stFileUploader {
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 20px !important;
        color: white !important;
    }

    /* Zurück-Button */
    .back-btn button {
        height: auto !important;
        width: auto !important;
        padding: 10px 25px !important;
        font-size: 1rem !important;
        border-radius: 15px !important;
        background: rgba(0,0,0,0.6) !important;
    }

    h1 { font-size: 7rem !important; letter-spacing: -4px; margin-bottom: 0px; }
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
    # Probiert es 3 Mal, falls der Server anonyme Anfragen verzögert
    for attempt in range(3):
        try:
            response = requests.post(API_URL, data=image_bytes, timeout=10)
            result = response.json()
            
            # Falls Modell lädt
            if isinstance(result, dict) and "estimated_time" in result:
                time.sleep(3)
                continue
            return result
        except:
            time.sleep(1)
    return None

# --- SEITE: HOME ---
if st.session_state['page'] == 'home':
    st.markdown("<br><br><h1>iFound</h1><p style='font-size: 1.5rem;'>Das intelligente Fundbüro im Wald.</p><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 2, 0.5])
    with col2:
        if st.button("🔎\n Etwas gefunden?"):
            set_page('melden')
            st.rerun()
    with col4:
        if st.button("📂\n Archiv öffnen"):
            set_page('archiv')
            st.rerun()

# --- SEITE: MELDEN ---
elif st.session_state['page'] == 'melden':
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Zurück zum Dashboard"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2>Gegenstand registrieren</h2>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        uploaded_file = st.file_uploader("Bild auswählen oder hierher ziehen", type=["jpg", "jpeg", "png"])
        ort = st.text_input("Fundort eingeben...")

    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes))
        with c_right:
            st.image(image, width=350)
            if st.button("✨ KI-Analyse starten"):
                with st.spinner("Detektiv KI wird aktiv..."):
                    res = query_ki(img_bytes)
                    if res and isinstance(res, list):
                        label = res[0]['label'].split(",")[0].capitalize()
                        st.session_state['temp_item'] = label
                        st.success(f"Gefundenes Objekt: {label}")
                    else:
                        st.error("KI ist gerade überlastet. Bitte drück den Button nochmal!")

            if 'temp_item' in st.session_state:
                if st.button("🚀 Fall abschließen & Speichern"):
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

# --- SEITE: ARCHIV ---
elif st.session_state['page'] == 'archiv':
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Zurück zum Dashboard"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2>Alle registrierten Fundstücke</h2>", unsafe_allow_html=True)
    if not st.session_state['inventar']:
        st.info("Keine Fundstücke vorhanden.")
    else:
        for item in reversed(st.session_state['inventar']):
            with st.expander(f"📦 {item['name']} | {item['zeit']}"):
                col_img, col_info = st.columns([1, 2])
                col_img.image(item['bild'], width=200)
                col_info.write(f"**Ort des Fundes:** {item['ort']}")
                col_info.button("Abgeholt", key=f"btn_{item['zeit']}")
