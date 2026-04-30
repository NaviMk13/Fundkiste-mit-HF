import streamlit as st
import requests
import io
from PIL import Image
import datetime

# --- DESIGN & APPLE-STYLE INTERACTION ---
st.set_page_config(page_title="iFound | Next Gen", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    /* Große Klick-Flächen (Cards) */
    .big-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        padding: 60px 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        margin-bottom: 10px;
    }

    /* Versteckter Button-Trick: Wir stylen den Streamlit-Button so, 
       dass er die ganze Karte überlagert */
    div.stButton > button {
        height: 200px;
        width: 100% !important;
        background-color: transparent !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 30px !important;
        font-size: 1.5rem !important;
        font-weight: 200 !important;
        transition: all 0.3s !important;
    }

    div.stButton > button:hover {
        background-color: rgba(255,255,255,0.1) !important;
        transform: scale(1.02);
        border: 1px solid white !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    h1 {
        font-family: -apple-system, sans-serif;
        letter-spacing: -2px;
        font-weight: 700;
        font-size: 5rem !important;
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

# --- SEITE: HOME ---
if st.session_state['page'] == 'home':
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>iFound</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.6); font-size: 1.5rem;'>Das intelligente Fundbüro der Zukunft.</p>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Hier kommen die riesigen Auswahl-Flächen
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 0.2, 2, 0.5])
    
    with col2:
        st.markdown("<p style='text-align: center; color: white; margin-bottom: -50px; font-size: 4rem;'>🔍</p>", unsafe_allow_html=True)
        if st.button("Fund melden"):
            set_page('melden')
            st.rerun()
            
    with col4:
        st.markdown("<p style='text-align: center; color: white; margin-bottom: -50px; font-size: 4rem;'>📂</p>", unsafe_allow_html=True)
        if st.button("Archiv öffnen"):
            set_page('archiv')
            st.rerun()

# --- SEITE: MELDEN ---
elif st.session_state['page'] == 'melden':
    if st.button("← Zurück zum Dashboard"):
        set_page('home')
        st.rerun()
    
    st.markdown("<h2 style='color: white;'>Neuen Fund registrieren</h2>", unsafe_allow_html=True)
    # Hier kommt dein restlicher Code für den Upload...
    # (Siehe vorherige Nachricht für die KI-Logik)
    st.info("Hier kannst du jetzt dein Foto hochladen.")

# --- SEITE: ARCHIV ---
elif st.session_state['page'] == 'archiv':
    if st.button("← Zurück zum Dashboard"):
        set_page('home')
        st.rerun()
    
    st.markdown("<h2 style='color: white;'>Durchsuche alle Beweismittel</h2>", unsafe_allow_html=True)
    # Hier kommt dein restlicher Code für die Liste...
    st.write("Aktuell sind 0 Gegenstände im System.")
