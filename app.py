import streamlit as st
from transformers import pipeline
from PIL import Image
import datetime

# --- 1. KONFIGURATION & SPEICHER ---
st.set_page_config(page_title="Schul-Fundbüro", layout="wide")

if 'fund_liste' not in st.session_state:
    st.session_state['fund_liste'] = []

# KI-Modell laden
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# --- 2. HAUPTSEITE & NAVIGATION ---
st.title("🏫 Digitales Schul-Fundbüro")

# Tabs für bessere Übersicht
tab1, tab2 = st.tabs(["➕ Neuen Fund melden", "🔍 Fundstücke suchen"])

# --- TAB 1: NEUEN FUND MELDEN ---
with tab1:
    st.header("Neuen Gegenstand registrieren")
    
    col_input, col_preview = st.columns(2)
    
    with col_input:
        quelle = st.radio("Bildquelle wählen:", ["Kamera", "Datei-Upload"])
        
        img_file = None
        if quelle == "Kamera":
            img_file = st.camera_input("Foto machen")
        else:
            img_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])
        
        beschreibung = st.text_area("Zusätzliche Beschreibung", placeholder="z.B. Farbe, Fundort, Besonderheiten...")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        
        with st.spinner("KI erkennt Objekt..."):
            results = classifier(image)
            ki_label = results[0]['label']
        
        with col_preview:
            st.image(image, caption="Vorschau", use_container_width=True)
            st.info(f"KI-Vorschlag: **{ki_label}**")
            
            if st.button("Offiziell eintragen"):
                neuer_eintrag = {
                    "id": len(st.session_state['fund_liste']) + 1,
                    "name": ki_label,
                    "beschreibung": beschreibung,
                    "bild": image,
                    "datum": datetime.date.today().strftime("%d.%m.%Y")
                }
                st.session_state['fund_liste'].insert(0, neuer_eintrag)
                st.success("Gegenstand wurde im System gespeichert!")
                st.balloons()

# --- TAB 2: SUCHFUNKTION & LISTE ---
with tab2:
    st.header("Fundstücke durchsuchen")
    
    suchbegriff = st.text_input("Suchen nach Name oder Beschreibung...", "").lower()
    
    # Filter-Logik
    gefilterte_liste = [
        item for item in st.session_state['fund_liste']
        if suchbegriff in item['name'].lower() or suchbegriff in item['beschreibung'].lower()
    ]
    
    if not gefilterte_liste:
        st.warning("Keine passenden Fundstücke gefunden.")
    else:
        # Anzeige in einem Grid (Kacheln)
        cols = st.columns(3)
        for i, item in enumerate(gefilterte_liste):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(item['bild'], use_container_width=True)
                    st.subheader(item['name'])
                    st.write(f"📅 **Datum:** {item['datum']}")
                    st.write(f"📝 **Info:** {item['beschreibung']}")
