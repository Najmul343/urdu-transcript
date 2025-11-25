import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

# ---------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Urdu Transcriber",
    page_icon="📝",
    layout="centered"
)

st.title("🇵🇰 Urdu Audio/Video Transcriber")
st.markdown("### WhatsApp voice → Beautiful Urdu text")
st.caption("2025 • CPU Optimized • Cloud Ready")

# ---------------------------------------------------------
# Load Whisper Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return WhisperModel(
        "tiny",   # safest for cloud
        device="cpu",
        compute_type="int8"
    )

model = load_model()
st.success("Model loaded successfully ✔️")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
file = st.file_uploader(
    "Upload Audio or Video File",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"]
)

# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------
if file:
    st.audio(file)

    if st.button("Transcribe to Urdu"):
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            path = tmp.name

        with st.spinner("Transcribing..."):
            segments, info = model.transcribe(
                path, language="ur", beam_size=5
            )
            text = " ".join([seg.text for seg in segments])

        os.remove(path)

        st.success("Transcription Complete ✔️")
        st.markdown("### Urdu Transcript:")
        st.write(text)

        st.download_button(
            "Download Urdu Text",
            text,
            file_name="urdu_transcript.txt"
        )

        st.balloons()

else:
    st.info("Please upload an audio/video file.")
