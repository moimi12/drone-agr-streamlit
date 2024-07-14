import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import time
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Define class labels for the model predictions
class_labels = {
    0: "malade",
    1: "saine"
}

# Example of previous simulation reports with drones
simulation_reports = [
    {"date": "2024-07-12", "location": "35.193581° N, -0.610751° W", "prédiction": 'malade'},
    {"date": "2024-07-11", "location": "35.218719° N, -0.634968° W", "prédiction": 'malade'},
    {"date": "2024-07-11", "location": "35.218681° N, -0.634990° W", "prédiction": 'malade'}
]

def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #d4f4dd;
            color: #000000;
        }
        .stApp {
            background-color: #ffffff;
        }
        .stSidebar {
            background-color: #d4f4dd;
        }
        .stButton>button {
            background-color: #4caf50;
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #000000;
        }
        .stHeader, .stSubheader, .stMarkdown, .stText {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def previous_reports():
    st.subheader("Rapports précédents")
    
    if simulation_reports:
        for report in simulation_reports:
            st.write(f"Date : {report['date']}")
            st.write(f"Localisation GPS : {report['location']}")
            st.write(f"Prédiction de la maladie : {report['prédiction']}")
            st.write("---")
    else:
        st.write("Aucun rapport de simulation disponible.")

def program_mission():
    st.subheader("Programmer une mission")
    img = "map.jpeg"
    st.image(img)
    point_a = st.text_input("Point A (longitude, latitude, altitude)")
    point_b = st.text_input("Point B (longitude, latitude, altitude)")
    point_c = st.text_input("Point C (longitude, latitude, altitude)")
    point_d = st.text_input("Point D (longitude, latitude, altitude)")

    if st.button("Valider"):
        st.write("Mission validée !")

def predict_disease_single(image_path):
    model_path = "./Model-wheat/wheat.keras"
    model = load_model(model_path)
    
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        prediction = model.predict(x)[0]
        class_label = class_labels[int(prediction >= 0.5)]
        return class_label
    except FileNotFoundError:
        st.write(f"Fichier image non trouvé : {image_path}")
        return None

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.capture_interval = 2 * 30  # Assuming 30 FPS
        self.images_captured = False
        self.current_capture_time = 2
        self.image_path = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:  # Update every second
            if self.current_capture_time > 1:
                self.current_capture_time -= 1
            else:
                self.current_capture_time = 2

        if self.frame_count % self.capture_interval == 0:
            self.image_path = f"capture_{self.frame_count // self.capture_interval}.jpg"
            cv2.imwrite(self.image_path, img)
            self.images_captured = True

        return img

def simulate_drone():
    st.subheader("Simulation avec le drone")
    
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
    )

    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        placeholder = st.empty()
        start_button = st.button("Démarrer la simulation")
        
        if start_button:
            webrtc_ctx.video_processor.frame_count = 0
            webrtc_ctx.video_processor.images_captured = False
            processing = True

            start_time = time.time()
            location = st.text_input("Entrez la localisation GPS (longitude, latitude)")

            while processing:
                if webrtc_ctx.video_processor.images_captured:
                    image_path = webrtc_ctx.video_processor.image_path
                    if image_path:
                        class_label = predict_disease_single(image_path)
                        img = cv2.imread(image_path)
                        if class_label:
                            st.image(img, caption=f"Image capturée : {image_path} - {class_label}", width=300)
                            st.write(f"Résultat de la prédiction : {class_label}")
                        webrtc_ctx.video_processor.images_captured = False
                    time.sleep(2)
                else:
                    time.sleep(1)

            end_time = time.time()
            duration = end_time - start_time
            minutes, seconds = divmod(duration, 60)
            formatted_duration = f"{int(minutes)} minutes {int(seconds)} seconds"

            # Ajouter un rapport de simulation
            if location:
                simulation_reports.append({
                    "date": time.strftime("%Y-%m-%d", time.localtime()),
                    "location": location,
                    "duration": formatted_duration
                })

def main():
    # Apply custom CSS
    apply_custom_css()
    
    
    # Interface d'accueil
    st.sidebar.image("logo.png", use_column_width=True)  # Assurez-vous que le logo.png est dans votre répertoire de travail
    st.sidebar.title("AGROGARD-TECH")
    st.sidebar.write(" AGROGARD-TECH, une technologie intelligente pour la surveillance agricole à base de drones. Utilisez les options ci-dessous pour explorer les fonctionnalités de notre service.")

    option = st.sidebar.selectbox("Sélectionnez une option", ("Accueil", "Rapports précédents", "Programmer une mission", "Simulation local", "Simulation avec le drone"))
    
    if option == "Accueil":
        st.write("### Bienvenue sur AGROGARD-TECH")
        st.write("""
        AGROGARD-TECH est une technologie intelligente pour la surveillance des cultures agricoles à base de drones équipés de caméras, de modèles de deep learning et AI, nous offrons des solutions avancées pour :
        - Surveiller les champs en temps réel
        - Programmer des missions de surveillance
        - Détecter les maladies agricoles

        Explorez les différentes fonctionnalités à l'aide de la barre latérale.
        """)
    elif option == "Rapports précédents":
        previous_reports()
    elif option == "Programmer une mission":
        program_mission()
    elif option == "Simulation local":
        image_paths = []
        for i in range(1, 5):
            image_path = st.text_input(f"Chemin de l'image {i}")
            if image_path:
                image_paths.append(image_path)
        
        if len(image_paths) == 4:
            for img_path in image_paths:
                class_label = predict_disease_single(img_path)
                if class_label:
                    st.write(f"Image {img_path} - Résultat de la prédiction : {class_label}")
    elif option == "Simulation avec le drone":
        simulate_drone()

if __name__ == "__main__":
    main()