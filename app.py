import streamlit as st
import requests
import io
from PIL import Image
import datetime

# --- KONFIGURATION & DESIGN ---
st.set_page_config(page_title="iFound | Nature Edition", layout="wide", initial_sidebar_state="collapsed")

# Dein Hugging Face Token (optional)
HEADERS = {"Authorization": "Bearer DEIN_TOKEN_HIER"} 
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

st.markdown("""
    <style>
    .stApp {
        /* Zurück zum Wald-Hintergrund */
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    /* Große Kacheln auf dem Home-Bildschirm */
    div.stButton > button {
        height: 220px;
        width: 100% !important;
        background-color: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(12px);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 40px !important;
        font-size: 1.8rem !important;
        font-weight: 300 !important;
        transition: all 0.4s ease !important;
    }

    div.stButton > button:hover {
        background-color: rgba(255,255,255,0.2) !important;
        transform: translateY(-10px);
        border: 1px solid white !important;
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }

    /* Spezielles Styling für den Datei-Upload Bereich */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 20px;
        border: 1px dashed rgba(255,255,255,0.4);
    }

    h1 { font-family: -apple-system, sans-serif; font-weight: 800; font-size: 5rem !important; color: white; text-align: center; margin-bottom: 0px; }
    p { color: rgba(255,255,255,0.8); text-align: center; font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'inventar' not in st.session_state:
    st.session_state['inventar'] = []

def set_page(name):
    st.session_state['page'] = name

def query_ki(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
    return response.json()

# --- NAVIGATION ---

# 1. HOME SEITE
if st.session_state['page'] == 'home':
    st.markdown("<br><br><br><h1>iFound</h1><p>Ehrlichkeit, die sich natürlich anfühlt.</p><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 2, 0.5])
    with col2:
        if st.button("🔎\nFund melden"):
            set_page('melden')
            st.rerun()
    with col4:
        if st.button("📂\nArchiv öffnen"):
            set_page('archiv')
            st.rerun()

# 2. MELDEN SEITE
elif st.session_state['page'] == 'melden':
    if st.button("← Abbrechen"):
        set_page('home')
        st.rerun()

    st.markdown("<h2 style='color: white;'>Schritt 1: Beweismittel sichern</h2>", unsafe_allow_html=True)
    
    col_up, col_pre = st.columns(2)
    
    with col_up:
        # Hier ist dein Button zum Öffnen der Dateien
        uploaded_file = st.file_uploader("Klicke hier oder ziehe ein Bild hinein", type=["jpg", "jpeg", "png"])
        ort = st.text_input("Fundort", placeholder="Wo hast du es entdeckt?")

    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes))
        
        with col_pre:
            st.image(image, caption="Vorschau des Fundstücks", width=300)
            if st.button("✨ KI-Experten fragen"):
                with st.spinner("Die KI blättert im Katalog..."):
                    try:
                        res = query_ki(img_bytes)
                        label = res[0]['label'].split(",")[0]
                        st.session_state['temp_item'] = label
                        st.success(f"Gefunden: {label}")
                    except:
                        st.error("Der KI-Server antwortet gerade nicht.")
            
            if 'temp_item' in st.session_state:
                if st.button("✅ Offiziell registrieren"):
                    st.session_state['inventar'].append({
                        "name": st.session_state['temp_item'],
                        "ort": ort,
                        "bild": image,
                        "zeit": datetime.datetime.now().strftime("%d.%m. %H:%M")
                    })
                    st.balloons()
                    del st.session_state['temp_item']
                    set_page('home')
                    st.rerun()

# 3. ARCHIV SEITE
elif st.session_state['page'] == 'archiv':
    if st.button("← Zurück zum Dashboard"):
        set_page('home')
        st.rerun()

    st.markdown("<h2 style='color: white;'>Eingelagerte Gegenstände</h2>", unsafe_allow_html=True)
    
    if not st.session_state['inventar']:
        st.info("Noch keine Gegenstände im Archiv.")
    else:
        for item in reversed(st.session_state['inventar']):
            with st.expander(f"📦 {item['name']} (Gefunden: {item['zeit']})"):
                c1, c2 = st.columns([1, 2])
                c1.image(item['bild'], width=150)
                c2.write(f"**Ort:** {item['ort']}")
                c2.write("Status: *Wartet auf den Besitzer*")
