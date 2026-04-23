import streamlit as st
from transformers import pipeline
from PIL import Image
import datetime

# --- 1. SETUP & SPEICHER ---
st.set_page_config(page_title="Schul-Fundbüro KI", layout="wide")

# Initialisiere den Speicher für Fundstücke
if 'fund_liste' not in st.session_state:
    st.session_state['fund_liste'] = []

# KI-Modell von Hugging Face laden (wird gecacht)
@st.cache_resource
def load_hf_model():
    # Dieses Modell ist ein Profi-Bildklassifizierer von Google
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_hf_model()

# --- 2. NAVIGATION ---
st.title("🏫 Digitales Schul-Fundbüro 2.0")
tab1, tab2 = st.tabs(["➕ Fund melden", "🔍 Fundstücke suchen"])

# --- TAB 1: FUND MELDEN ---
with tab1:
    st.header("Neuen Gegenstand registrieren")
    
    col_input, col_preview = st.columns(2)
    
    with col_input:
        quelle = st.radio("Bildquelle:", ["Datei hochladen", "Kamera"])
        
        if quelle == "Kamera":
            img_file = st.camera_input("Foto machen")
        else:
            img_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])
        
        user_beschreibung = st.text_area("Details zum Fund", placeholder="z.B. Gefunden im Chemieraum, blaue Hülle...")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        
        with st.spinner("Hugging Face KI analysiert das Bild..."):
            # Die KI erkennt das Objekt
            predictions = classifier(image)
            top_prediction = predictions[0]['label']
            confidence = predictions[0]['score']
        
        with col_preview:
            st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
            st.info(f"KI-Vorschlag: **{top_prediction}** ({confidence*100:.1f}% sicher)")
            
            if st.button("In Fundkiste speichern"):
                neuer_eintrag = {
                    "name": top_prediction,
                    "beschreibung": user_beschreibung,
                    "bild": image,
                    "datum": datetime.date.today().strftime("%d.%m.%Y")
                }
                st.session_state['fund_liste'].insert(0, neuer_eintrag)
                st.success("Erfolgreich gespeichert!")
                st.balloons()

# --- TAB 2: SUCHEN & ANZEIGEN ---
with tab2:
    st.header("Alle Fundstücke durchsuchen")
    
    suche = st.text_input("Suchen nach Name oder Beschreibung...").lower()
    
    # Filter-Logik
    gefilterte_funde = [
        f for f in st.session_state['fund_liste']
        if suche in f['name'].lower() or suche in f['beschreibung'].lower()
    ]
    
    if not gefilterte_funde:
        st.info("Keine Fundstücke gefunden.")
    else:
        # Anzeige in 3 Spalten
        cols = st.columns(3)
        for i, fund in enumerate(gefilterte_funde):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(fund['bild'], use_container_width=True)
                    st.subheader(fund['name'])
                    st.write(f"📅 {fund['datum']}")
                    if fund['beschreibung']:
                        st.write(f"📝 {fund['beschreibung']}")
