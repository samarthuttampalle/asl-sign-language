import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
import os
import json

# --- Page Configuration ---
st.set_page_config(page_title="ASL Real-time Detection", layout="wide")

# --- ASL Labels ---
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'del', 'nothing', 'space']

# --- Core Application Class ---
class ASLDetector:
    def __init__(self, model_path):
        """Initialize the ASL detector with trained model"""
        self.model = None
        self.model_path = model_path
        self.input_height = 224
        self.input_width = 224
        self.expected_channels = 3
        self.index_to_label = {}
        
        self.load_model()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

    def load_model(self):
        # This function remains unchanged from the version that successfully loaded your model.
        # It's included here for completeness.
        try:
            tf.keras.backend.clear_session()
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            error_msg = str(e)
            is_shape_error = "stem_conv" in error_msg and "expected axis -1" in error_msg
            if is_shape_error:
                num_classes = len(ASL_LABELS)
                if not self.recreate_model_architecture(num_classes=num_classes):
                    raise RuntimeError("Failed to recreate model architecture.")
                self.model.load_weights(self.model_path)
            else:
                raise e # Re-raise other errors
        
        if self.model:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            input_shape = self.model.input_shape
            _, h, w, c = input_shape
            self.input_height = 224 if h is None else int(h)
            self.input_width = 224 if w is None else int(w)
            self.expected_channels = 3

    def preprocess_image(self, image, debug=False):
        # This function is unchanged.
        try:
            if image is None or image.size == 0: raise ValueError("Empty image")
            img_size = (self.input_width, self.input_height)
            if len(image.shape) == 2: image = np.stack([image]*3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1: image = np.concatenate([image]*3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = np.expand_dims(image, axis=0)
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input(image)
        except Exception:
            return np.zeros((1, self.input_height, self.input_width, 3), dtype=np.float32)

    def predict_sign(self, image, debug=False):
        # This function is unchanged.
        if self.model is None: return "Model not loaded", 0.0
        try:
            processed_image = self.preprocess_image(image, debug=debug)
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            predicted_label = ASL_LABELS[predicted_class] if predicted_class < len(ASL_LABELS) else "Unknown"
            return predicted_label, confidence
        except Exception as e:
            return f"Prediction error: {e}", 0.0

    def recreate_model_architecture(self, num_classes: int = 29):
        # This function is unchanged.
        try:
            tf.keras.backend.clear_session()
            base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
            base_model.trainable = False
            self.model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(num_classes, activation='softmax')])
            return True
        except: return False

    def load_labels(self, labels_path: str | None = None):
        # This function is unchanged.
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f: self.index_to_label = {int(k): v for k, v in json.load(f).items()}

    def speak_text(self, text):
        # This function is unchanged.
        try:
            def speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
        except Exception as e:
            st.error(f"TTS Error: {e}")

# --- Helper Function for Hand Detection ---
def extract_hand_roi(frame, use_center_crop=False):
    # This function is unchanged.
    if use_center_crop:
        h, w = frame.shape[:2]
        size = min(h, w) * 2 // 3
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
        roi = frame[y1:y1+size, x1:x1+size]
        cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (0, 255, 0), 2)
        return roi, frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 30
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(frame.shape[1] - x, w + 2*padding), min(frame.shape[0] - y, h + 2*padding)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return frame[y:y+h, x:x+w], frame
    return None, frame

# --- Main Streamlit Application ---
def main():
    st.title("🤟 ASL Real-time Detection System")
    st.markdown("Real-time American Sign Language detection with speech output.")

    # --- Session State Initialization ---
    if "run_camera" not in st.session_state: st.session_state.run_camera = False
    if "detected_text" not in st.session_state: st.session_state.detected_text = ""
    if "last_added_sign" not in st.session_state: st.session_state.last_added_sign = ""
    if "detector" not in st.session_state: st.session_state.detector = None

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("Model Path", value="sign_language_model_fixed.keras")
        labels_path = st.text_input("Labels JSON Path (optional)", value="class_labels.json")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.05)
        detection_mode = st.selectbox("Detection Mode", ["Auto (Skin Detection)", "Center Crop", "Full Frame"])
        
        # --- Robust Detector Initialization ---
        if st.button("🔄 Load / Reload Model"):
            try:
                with st.spinner("Loading model and TTS engine..."):
                    st.session_state.detector = ASLDetector(model_path)
            except Exception as e:
                st.error(f"Initialization Failed: {e}", icon="🚨")
                st.session_state.detector = None
        
        if st.session_state.detector and st.session_state.detector.model:
            st.success("✅ Model and TTS Engine are ready!")
        else:
            st.warning("⚠️ Please load the model to begin.")

    # --- Main UI (Conditional on successful initialization) ---
    if not st.session_state.detector:
        st.warning("Application not ready. Please click 'Load / Reload Model' in the sidebar.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📹 Camera Feed")
            if st.button("🎥 Start Camera", type="primary"): st.session_state.run_camera = True
            if st.button("⏹️ Stop Camera"): st.session_state.run_camera = False; st.rerun()
            video_placeholder = st.empty()

        with col2:
            st.subheader("🔤 Predictions")
            prediction_placeholder = st.empty()
            confidence_placeholder = st.empty()
            st.subheader("📝 Detected Text")
            st.text_area("Accumulated Text", value=st.session_state.detected_text, height=150, key="text_area")
            if st.button("🔊 Speak Full Text", type="primary"): st.session_state.detector.speak_text(st.session_state.detected_text)
            if st.button("🗑️ Clear Text"): st.session_state.detected_text = ""; st.rerun()

        # --- Camera Processing Loop ---
        if st.session_state.run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera.", icon="📷")
                st.session_state.run_camera = False
            else:
                stable_threshold, add_cooldown = 5, 1.5
                last_prediction_time, last_stable_sign, stable_counter = time.time(), "", 0
                try:
                    while st.session_state.run_camera:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        hand_roi, annotated_frame = extract_hand_roi(frame, detection_mode == "Center Crop")
                        current_sign, current_confidence = "...", 0.0
                        if hand_roi is not None and hand_roi.size > 0:
                            predicted_sign, confidence = st.session_state.detector.predict_sign(hand_roi)
                            if confidence >= confidence_threshold:
                                current_sign, current_confidence = predicted_sign, confidence
                                if current_sign == last_stable_sign and current_sign != 'nothing': stable_counter += 1
                                else: last_stable_sign, stable_counter = current_sign, 1
                                if stable_counter >= stable_threshold and (time.time() - last_prediction_time > add_cooldown):
                                    if current_sign != st.session_state.last_added_sign:
                                        if current_sign == 'space': st.session_state.detected_text += ' '; st.session_state.detector.speak_text("Space")
                                        elif current_sign == 'del': st.session_state.detected_text = st.session_state.detected_text[:-1]; st.session_state.detector.speak_text("Delete")
                                        else: st.session_state.detected_text += current_sign; st.session_state.detector.speak_text(current_sign)
                                        st.session_state.last_added_sign = current_sign
                                        last_prediction_time = time.time()
                        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                        prediction_placeholder.success(f"**Prediction: {current_sign}**" if current_confidence >= confidence_threshold else f"Prediction: {current_sign}")
                        with confidence_placeholder.container():
                            st.metric("Confidence", f"{current_confidence:.2%}")
                            st.progress(min(stable_counter / stable_threshold, 1.0))
                finally:
                    cap.release()

if __name__ == "__main__":
    main()