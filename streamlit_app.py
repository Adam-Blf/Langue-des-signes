import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
from lsf_model import LSFDetector
from letters_conditions import detect_letter_rules

st.set_page_config(page_title="LSF Detector", page_icon="🤟")

st.title("🤟 Langue des Signes Française - Detector")
st.write("Upload an image of a hand sign to detect the letter.")

# Load Model
@st.cache_resource
def load_model():
    model_path = "machine_learning/model.p"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                return data['model']
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_model()
detector = LSFDetector()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert to CV2 format
    img_array = np.array(image)
    
    # Process
    results, processed_image = detector.process_frame(img_array)
    
    # Draw landmarks
    processed_image = detector.draw_landmarks(processed_image, results)
    st.image(processed_image, caption='Processed Image', use_column_width=True)
    
    # Extract Keypoints
    keypoints = detector.extract_keypoints(results)
    
    detected_letter = None
    method = ""
    
    if results and results.multi_hand_landmarks:
        # Rule-based
        detected_letter = detect_letter_rules(results.multi_hand_landmarks[0])
        if detected_letter:
            method = "Rule-Based"
        
        # ML Fallback
        if not detected_letter and model:
            if np.any(keypoints):
                prediction = model.predict([keypoints])[0]
                probs = model.predict_proba([keypoints])[0]
                confidence = np.max(probs)
                
                if confidence > 0.6:
                    detected_letter = prediction
                    method = f"ML ({int(confidence * 100)}%)"
    
    if detected_letter:
        st.success(f"Detected Letter: {detected_letter} ({method})")
    else:
        st.warning("No letter detected or confidence too low.")
