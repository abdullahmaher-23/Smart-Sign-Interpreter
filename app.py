import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# 1. Page Configuration & Title
st.set_page_config(page_title="Smart Sign Interpreter", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #2e4053;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Smart Sign Interpreter")
st.sidebar.header("Project Dashboard")
st.sidebar.info("This system interprets hand gestures into meaningful phrases using Random Forest Classification.")


# 2. Load the trained model
@st.cache_resource
def load_trained_model():
    try:
        with open('.venv/model.p', 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    except FileNotFoundError:
        return None


model = load_trained_model()

if model is not None:
    st.sidebar.success("✅ Model: Loaded Successfully")
else:
    st.sidebar.error("❌ model.p not found! Please train the model first.")

# 3. MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# 4. Updated Labels based on your code
labels_dict = {0: 'Be Happy', 1: 'Watching You', 2: 'Awesome'}

# 5. Interface Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Hand Tracking")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Interpreter Output")
    result_placeholder = st.empty()
    st.write("---")
    st.write("**Detected Landmarks:**")
    landmark_status = st.empty()

# 6. Camera Stream Logic
run = st.checkbox('Start Interpreter', value=True)
cap = cv2.VideoCapture(1)  # Using index 1 as in your code

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Trying to access camera...")
        cap = cv2.VideoCapture(0)  # Fallback to 0 if 1 fails
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        landmark_status.success("Hand Detected")
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing landmarks
            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Data Processing
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prediction Logic
        if data_aux:
            try:
                # Making sure the model is fed with the correct feature size
                prediction = model.predict([np.asarray(data_aux)])
                predicted_label = labels_dict[int(prediction[0])]

                # Dynamic Display
                result_placeholder.markdown(
                    f"<div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: white;'>"
                    f"<h2 style='text-align: center; color: #333;'>Result:</h2>"
                    f"<h1 style='text-align: center; color: #4CAF50; font-size: 60px;'>{predicted_label}</h1>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            except:
                pass
    else:
        landmark_status.warning("No Hand Detected")
        result_placeholder.markdown("<h3 style='text-align: center; color: gray;'>Waiting for Gesture...</h3>",
                                    unsafe_allow_html=True)

    # Display the final image
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

cap.release()