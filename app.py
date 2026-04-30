import streamlit as st
import requests
import io
from PIL import Image
import random
import datetime

# --- DESIGN & ANIMATIONEN ---
st.set_page_config(page_title="iFound | KI Detektiv", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Hintergrund mit sanftem Zoom-Effekt */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica;
    }

    /* Apple-Style Karten-Animation */
    .nav-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .nav-card:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Text-Styling */
    h1, h2, h3, p {
        color: white !important;
        font-weight: 300;
    }

    /* Button-Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 15px;
        background: rgba(255,255,255,0.9);
        color: black;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIK ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'inventar' not in st.session_state:
    st.session_state['inventar'] = []

def set_page(name):
    st.session_state['page'] = name

# --- KI SETUP ---
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
HEADERS = {"Authorization": "Bearer DEIN_TOKEN"} # Optional

def query_ki(image_bytes):
    response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
    return response.json()

# --- STARTBILDSCHIRM (LANDING PAGE) ---
if st.session_state['page'] == 'home':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 4rem;'>iFound.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem;'>Ehrlichkeit trifft Intelligenz.</p>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="nav-card"><h3>🔍</h3><p>Etwas gefunden?</p></div>', unsafe_allow_html=True)
            if st.button("Fund melden"):
                set_page('melden')
                st.rerun()
        
        with c2:
            st.markdown('<div class="nav-card"><h3>📂</h3><p>Etwas verloren?</p></div>', unsafe_allow_html=True)
            if st.button("Archiv durchsuchen"):
                set_page('archiv')
                st.rerun()

# --- SEITE: FUND MELDEN ---
elif st.session_state['page'] == 'melden':
    if st.button("← Zurück"):
        set_page('home')
        st.rerun()
        
    st.header("Neuen Fall anlegen")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Beweisfoto hochladen", type=["jpg", "png", "jpeg"])
        ort = st.text_input("Fundort")
    
    if uploaded_file:
        img_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(img_bytes))
        
        with col_b:
            st.image(image, width=350)
            if st.button("KI-Analyse starten"):
                with st.spinner("Analysiere..."):
                    try:
                        res = query_ki(img_bytes)
                        item_name = res[0]['label'].split(",")[0]
                        st.session_state['temp_item'] = item_name
                        st.info(f"🤖 'Das sieht für mich aus wie ein **{item_name}**. Ziemlich cooles Teil!'")
                    except:
                        st.error("Verbindung zum KI-Server unterbrochen.")
            
            if 'temp_item' in st.session_state:
                if st.button("Fall im Archiv speichern"):
                    st.session_state['inventar'].append({
                        "name": st.session_state['temp_item'],
                        "ort": ort,
                        "bild": image,
                        "zeit": datetime.datetime.now().strftime("%H:%M - %d.%m.%y")
                    })
                    st.balloons()
                    set_page('home')
                    del st.session_state['temp_item']
                    st.rerun()

# --- SEITE: ARCHIV ---
elif st.session_state['page'] == 'archiv':
    if st.button("← Zurück"):
        set_page('home')
        st.rerun()
        
    st.header("Das digitale Archiv")
    suche = st.text_input("Suche nach Gegenständen...")
    
    if not st.session_state['inventar']:
        st.write("Das Archiv ist noch leer.")
    else:
        for i, item in enumerate(reversed(st.session_state['inventar'])):
            if suche.lower() in item['name'].lower() or suche.lower() in item['ort'].lower():
                with st.expander(f"{item['name']} | {item['zeit']}"):
                    c1, c2 = st.columns([1, 3])
                    c1.image(item['bild'], width=150)
                    c2.write(f"**Fundort:** {item['ort']}")
                    if c2.button(f"Abgeholt markieren", key=f"del_{i}"):
                        st.session_state['inventar'].pop() # Vereinfacht für das Beispiel
                        st.rerun()
