"""
Sign Language & Emotion Detection — Streamlit App
===================================================
Real-time webcam feed that detects:
  • Hand sign-language gestures  (WLASL model)
  • Facial emotions              (FER-2013 model)

Uses streamlit-webrtc for browser-based video capture and
MediaPipe + OpenCV for pre-processing.

MODEL FILES (not shipped):
  - sign_language_model.h5   →  Trained on WLASL gesture classes
  - emotion_model.h5         →  Trained on FER-2013 emotion classes
Place them next to this file. If they are absent the app runs with
mock/random predictions so the UI can still be demonstrated.
"""

import os
import time
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ──────────────────────────────────────────────
# Constants & Labels
# ──────────────────────────────────────────────
SIGN_MODEL_PATH = Path(__file__).parent / "sign_language_model.h5"
EMOTION_MODEL_PATH = Path(__file__).parent / "emotion_model.h5"

SIGN_LABELS = [
    "hello", "thanks", "yes", "no", "please",
    "sorry", "help", "love", "friend", "good",
    "bad", "eat", "drink", "more", "stop",
    "go", "come", "want", "need", "like",
]

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral",
]

# Cooldown (seconds) before the same sign is appended again
SIGN_COOLDOWN = 2.0

# Confidence threshold for accepting a prediction
CONFIDENCE_THRESHOLD = 0.6

# ──────────────────────────────────────────────
# Model Loading (with graceful fallback)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    """Return (sign_model, emotion_model). Either may be None."""
    sign_model = None
    emotion_model = None
    try:
        from tensorflow.keras.models import load_model  # noqa: F401
        if SIGN_MODEL_PATH.exists():
            sign_model = load_model(str(SIGN_MODEL_PATH))
        if EMOTION_MODEL_PATH.exists():
            emotion_model = load_model(str(EMOTION_MODEL_PATH))
    except Exception as e:
        st.warning(f"Could not load one or both models: {e}")
    return sign_model, emotion_model


# ──────────────────────────────────────────────
# Thread-safe shared state
# ──────────────────────────────────────────────
class SharedState:
    """Thread-safe container shared between the WebRTC callback and Streamlit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._emotion = "—"
        self._sentence_parts: list[str] = []
        self._last_sign = ""
        self._last_sign_time = 0.0

    # --- emotion ---
    @property
    def emotion(self) -> str:
        with self._lock:
            return self._emotion

    @emotion.setter
    def emotion(self, value: str):
        with self._lock:
            self._emotion = value

    # --- sentence ---
    @property
    def sentence(self) -> str:
        with self._lock:
            return " ".join(self._sentence_parts)

    def append_sign(self, sign: str):
        now = time.time()
        with self._lock:
            if sign == self._last_sign and (now - self._last_sign_time) < SIGN_COOLDOWN:
                return  # debounce identical consecutive signs
            self._sentence_parts.append(sign)
            self._last_sign = sign
            self._last_sign_time = now

    def clear_sentence(self):
        with self._lock:
            self._sentence_parts.clear()
            self._last_sign = ""
            self._last_sign_time = 0.0


# Create a singleton shared state (persisted across reruns via session state)
if "shared" not in st.session_state:
    st.session_state["shared"] = SharedState()
shared: SharedState = st.session_state["shared"]


# ──────────────────────────────────────────────
# Video Processor
# ──────────────────────────────────────────────
class SignEmotionProcessor(VideoProcessorBase):
    """Processes each video frame for sign and emotion detection."""

    def __init__(self):
        self.sign_model, self.emotion_model = load_models()

        # MediaPipe Hands — for sign language landmark extraction
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Haar cascade for quick face detection (emotion ROI)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── helpers ──────────────────────────────
    def _extract_hand_landmarks(self, rgb_frame):
        """Return a flat numpy array of 21×3 hand landmarks (or None)."""
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                dtype=np.float32,
            ).flatten()
            return landmarks, results.multi_hand_landmarks
        return None, None

    def _predict_sign(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) from the sign language model."""
        if self.sign_model is not None:
            inp = landmarks.reshape(1, -1)
            preds = self.sign_model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            label = SIGN_LABELS[idx] if idx < len(SIGN_LABELS) else f"sign_{idx}"
            return label, conf
        else:
            # Mock prediction for demo
            idx = int(np.sum(landmarks * 1000)) % len(SIGN_LABELS)
            return SIGN_LABELS[idx], 0.75

    def _predict_emotion(self, gray_face: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) from the emotion model."""
        if self.emotion_model is not None:
            face_resized = cv2.resize(gray_face, (48, 48))
            inp = face_resized.astype("float32") / 255.0
            inp = inp.reshape(1, 48, 48, 1)
            preds = self.emotion_model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"emo_{idx}"
            return label, conf
        else:
            # Mock prediction
            idx = int(np.mean(gray_face)) % len(EMOTION_LABELS)
            return EMOTION_LABELS[idx], 0.80

    # ── main callback ───────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Sign Language Detection ---
        landmarks, multi_hand = self._extract_hand_landmarks(rgb)
        if landmarks is not None:
            sign_label, sign_conf = self._predict_sign(landmarks)

            # Draw hand landmarks
            for hand_lm in multi_hand:
                self.mp_draw.draw_landmarks(
                    img, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2),
                )

            if sign_conf >= CONFIDENCE_THRESHOLD:
                shared.append_sign(sign_label)
                # Overlay sign text
                cv2.putText(
                    img, f"Sign: {sign_label} ({sign_conf:.0%})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 128), 2, cv2.LINE_AA,
                )

        # --- Emotion Detection ---
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            emo_label, emo_conf = self._predict_emotion(face_roi)
            shared.emotion = f"{emo_label} ({emo_conf:.0%})"

            # Draw face rectangle & emotion label
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(
                img, emo_label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 200, 0), 2, cv2.LINE_AA,
            )
            break  # use the first detected face only

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sign Language & Emotion Detector",
        page_icon="🤟",
        layout="wide",
    )

    # ── Custom CSS ───────────────────────────
    st.markdown(
        """
        <style>
        /* Dark gradient background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.85);
            backdrop-filter: blur(12px);
        }
        /* Glassmorphism card */
        .glass-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 24px 28px;
            backdrop-filter: blur(14px);
            margin-bottom: 16px;
        }
        .glass-card h3 {
            margin-top: 0;
            color: #a78bfa;
        }
        .emotion-value {
            font-size: 2rem;
            font-weight: 700;
            color: #facc15;
            text-shadow: 0 0 14px rgba(250,204,21,.45);
        }
        .sentence-box {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 18px 22px;
            font-size: 1.25rem;
            color: #e2e8f0;
            min-height: 56px;
            letter-spacing: .3px;
            line-height: 1.6;
        }
        .app-title {
            text-align: center;
            background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .app-subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎭 Current Emotion")
        emotion_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ About")
        st.markdown(
            "This app uses **MediaPipe** for hand tracking, a **WLASL-trained** model "
            "for sign language recognition, and a **FER-2013-trained** model for facial "
            "emotion detection — all running in real-time in your browser."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Instructions")
        st.markdown(
            "1. Allow camera access when prompted\n"
            "2. Show hand signs to detect gestures\n"
            "3. Detected signs build a sentence below\n"
            "4. Your facial emotion updates in the sidebar"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Sentence", use_container_width=True):
            shared.clear_sentence()

    # ── Main Area ────────────────────────────
    st.markdown('<p class="app-title">🤟 Sign Language & Emotion Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Real-time gesture recognition &amp; emotion analysis</p>',
        unsafe_allow_html=True,
    )

    # Model status badges
    sign_model, emotion_model = load_models()
    col1, col2 = st.columns(2)
    with col1:
        status = "✅ Loaded" if sign_model else "⚠️ Demo Mode (no model file)"
        st.info(f"**Sign Model:** {status}")
    with col2:
        status = "✅ Loaded" if emotion_model else "⚠️ Demo Mode (no model file)"
        st.info(f"**Emotion Model:** {status}")

    # WebRTC streamer
    ctx = webrtc_streamer(
        key="sign-emotion",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignEmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # ── Live-updated panels (poll shared state) ──
    st.markdown("---")
    st.markdown("### 📝 Full Detected Sentence")
    sentence_placeholder = st.empty()

    # Polling loop — refreshes the sidebar emotion and the sentence box
    if ctx.state.playing:
        while True:
            emotion_placeholder.markdown(
                f'<p class="emotion-value">{shared.emotion}</p>',
                unsafe_allow_html=True,
            )
            sentence_text = shared.sentence or "_(signs will appear here as you gesture)_"
            sentence_placeholder.markdown(
                f'<div class="sentence-box">{sentence_text}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.3)
    else:
        emotion_placeholder.markdown(
            '<p class="emotion-value">—</p>', unsafe_allow_html=True,
        )
        sentence_text = shared.sentence or "_(start the webcam to begin detection)_"
        sentence_placeholder.markdown(
            f'<div class="sentence-box">{sentence_text}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
