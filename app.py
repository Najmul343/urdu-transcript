import streamlit as st
from pathlib import Path
import tempfile
import os
import google.generativeai as genai

st.set_page_config(page_title="UrduScribe AI")

# API key
api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key:
    st.error("⚠️ Add GEMINI_API_KEY in Streamlit Secrets.")
else:
    genai.configure(api_key=api_key)

st.title("UrduScribe AI – Audio to Text (Python Streamlit Version)")

uploaded_file = st.file_uploader("Upload audio (MP3, WAV, M4A, OGG)",
                                 type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file and st.button("Transcribe", type="primary"):
    with st.spinner("Transcribing… Please wait ⏳"):

        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load audio bytes
        audio_bytes = Path(temp_audio_path).read_bytes()

        # Correct audio input format for Gemini
        audio_part = {
            "mime_type": uploaded_file.type,
            "data": audio_bytes
        }

        # Create model
        model = genai.GenerativeModel("models/gemini-2.0-flash")

        # Send request
        result = model.generate_content(
            [audio_part, "Transcribe this Urdu audio to Urdu text."]
        )

        transcription = result.text

    st.success("Transcription completed ✔")
    st.subheader("📝 Urdu Transcription:")
    st.write(transcription)

    st.download_button("Download Transcript",
                       transcription,
                       file_name="urdu_transcript.txt",
                       mime="text/plain")
