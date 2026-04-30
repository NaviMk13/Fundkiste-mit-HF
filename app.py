import streamlit as st
import requests
import io
from PIL import Image
import datetime

# --- KONFIGURATION & DESIGN ---
st.set_page_config(page_title="iFound | Next Gen", layout="wide", initial_sidebar_state="collapsed")

# Dein Hugging Face Token (optional, aber empfohlen für Stabilität)
HEADERS = {"Authorization": "Bearer DEIN_TOKEN_HIER"} 
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    /* Große Kacheln auf dem Home-Bildschirm */
    div.stButton > button {
        height: 180px;
        width: 100% !important;
        background-color: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(15px);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 30px !important;
        font-size: 1.5rem !important;
        font-weight: 200 !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }

    div.stButton > button:hover {
        background-color: rgba(255,255,255,0.15) !important;
        transform: scale(1.05);
        border: 1px solid white !important;
    }

    /* Zurück-Button Styling */
    .back-btn button {
        height: auto !important;
        width: auto !important;
        padding: 10px 20px !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
    }

    h1 { font-family: -apple-system, sans-serif; font-weight: 700; font-size: 4rem !important; color: white; text-align: center; }
    p { color: rgba(255,255,255,0.7); text-align: center; }
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
    st.markdown("<br><br><h1>iFound</h1><p>Intelligentes Fundbüro</p><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 2, 0.5])
    with col2:
        if st.button("🔍\nFund melden"):
            set_page('melden')
            st.rerun()
    with col4:
        if st.button("📂\nArchiv öffnen"):
            set_page('archiv')
            st.rerun()

# 2. MELDEN SEITE
elif st.session_state['page'] == 'melden':
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Zurück"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2 style='color: white;'>Neuen Fall registrieren</h2>", unsafe_allow_html=True)
    
    col_up, col_pre = st.columns(2)
    
    with col_up:
        # HIER ist der Button zum Dateien öffnen!
        uploaded_file = st.file_uploader("Bild des Gegenstands wählen", type=["jpg", "jpeg", "png"])
        ort = st.text_input("Wo wurde es gefunden?", placeholder="z.B. Mensa, Raum 204")

    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes))
        
        with col_pre:
            st.image(image, caption="Vorschau", width=300)
            if st.button("✨ KI-Analyse starten"):
                with st.spinner("KI identifiziert Objekt..."):
                    try:
                        res = query_ki(img_bytes)
                        label = res[0]['label'].split(",")[0]
                        st.session_state['temp_item'] = label
                        st.success(f"Erkannt: {label}")
                    except:
                        st.error("Fehler bei der KI-Anfrage.")
            
            if 'temp_item' in st.session_state:
                if st.button("✅ Im Archiv speichern"):
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
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Zurück"):
        set_page('home')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2 style='color: white;'>Das digitale Archiv</h2>", unsafe_allow_html=True)
    
    if not st.session_state['inventar']:
        st.info("Das Archiv ist noch leer. Melde einen Fund, um es zu füllen!")
    else:
        for item in reversed(st.session_state['inventar']):
            with st.expander(f"{item['name']} - {item['zeit']}"):
                c1, c2 = st.columns([1, 2])
                c1.image(item['bild'], width=150)
                c2.write(f"**Ort:** {item['ort']}")
