import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Charger le classificateur Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Fonction pour détecter les visages sur une image
def detect_faces_on_image(frame, scaleFactor, minNeighbors, rect_color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
    return frame, faces

# Interface Streamlit
def app():
    st.title("Détection de visages avec l'algorithme Viola-Jones")
    st.write("Cette application détecte les visages via votre webcam.")
    st.write("Utilisez les options ci-dessous pour personnaliser la détection.")

    # Sliders et options
    scaleFactor = st.slider("Ajuster scaleFactor", 1.1, 2.0, 1.3, 0.05, key="slider_scaleFactor")
    minNeighbors = st.slider("Ajuster minNeighbors", 1, 10, 5, 1, key="slider_minNeighbors")
    rect_color_hex = st.color_picker("Choisir la couleur des rectangles", "#00FF00", key="color_picker_rect")
    save_image = st.checkbox("Sauvegarder l'image avec visages détectés", key="checkbox_save_image")

    # Convertir couleur hex -> BGR
    hex_color = tuple(int(rect_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    rect_color_bgr = (hex_color[2], hex_color[1], hex_color[0])

    # Capture d'image via webcam
    img_file_buffer = st.camera_input("Prenez une photo")

    if img_file_buffer is not None:
        # Convertir en array OpenCV
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Détecter visages
        frame_with_faces, faces = detect_faces_on_image(frame, scaleFactor, minNeighbors, rect_color_bgr)

        # Afficher image avec rectangles
        frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

        # Télécharger si demandé
        if save_image and len(faces) > 0:
            _, buffer = cv2.imencode('.png', frame_with_faces)
            st.download_button(
                label="Télécharger l'image avec visages",
                data=buffer.tobytes(),
                file_name=f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

if __name__ == "__main__":
    app()


