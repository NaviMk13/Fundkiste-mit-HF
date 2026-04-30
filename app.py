import streamlit as st
import requests, io, datetime
from PIL import Image

# --- SCHNELLES DESIGN ---
st.set_page_config(page_title="iFound", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }
    h1, h2, p, label { color: white !important; text-shadow: 2px 2px 5px black !important; }
    div.stButton > button {
        height: 150px; width: 100%; border-radius: 20px;
        background: rgba(255,255,255,0.2) !important; color: white !important;
        font-size: 1.5rem !important; transition: 0.3s;
    }
    div.stButton > button:hover { transform: translateY(-5px); background: rgba(255,255,255,0.3) !important; }
    </style>
    """, unsafe_allow_html=True)

# --- APP LOGIK ---
if 'page' not in st.session_state: st.session_state['page'] = 'home'
if 'inventar' not in st.session_state: st.session_state['inventar'] = []

# HOME
if st.session_state['page'] == 'home':
    st.markdown("<br><h1>iFound</h1><p>Schnell & Zuverlässig</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🔎 Fund melden"): st.session_state['page'] = 'melden'; st.rerun()
    if c2.button("📂 Archiv"): st.session_state['page'] = 'archiv'; st.rerun()

# MELDEN (Mit robustem Google-Modell)
elif st.session_state['page'] == 'melden':
    if st.button("← Zurück"): st.session_state['page'] = 'home'; st.rerun()
    
    file = st.file_uploader("Bild wählen", type=["jpg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, width=250)
        if st.button("✨ KI-Check"):
            with st.spinner("Analyse..."):
                # Wir probieren ein extrem stabiles Modell
                API = "https://api-inference.huggingface.co/models/google/mobilenet_v2_1.0_224"
                res = requests.post(API, data=file.getvalue())
                try:
                    name = res.json()[0]['label'].split(",")[0].capitalize()
                    st.success(f"Erkannt: {name}")
                    if st.button("Speichern"):
                        st.session_state['inventar'].append({"name": name, "zeit": datetime.datetime.now().strftime("%H:%M")})
                        st.session_state['page'] = 'home'; st.rerun()
                except:
                    st.error("KI antwortet nicht. Web-Limit erreicht.")

# ARCHIV
elif st.session_state['page'] == 'archiv':
    if st.button("← Zurück"): st.session_state['page'] = 'home'; st.rerun()
    for i in st.session_state['inventar']: st.write(f"📦 {i['name']} ({i['zeit']})")
