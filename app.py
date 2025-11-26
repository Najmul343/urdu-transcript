import streamlit as st
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os
import google.generativeai as genai

# -------------------------
#  PAGE SETTINGS
# -------------------------
st.set_page_config(
    page_title="UrduScribe AI",
    layout="wide",
)

st.markdown("""
    <style>
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to bottom right, #f0fdf4, #ffffff, #ecfdf5);
    }
    .urdu-text {
        font-family: "Noto Nastaliq Urdu", serif;
        font-size: 22px;
        line-height: 1.9;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
#  GEMINI API KEY
# -------------------------
api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key:
    st.warning("⚠️ Please add GEMINI_API_KEY to Streamlit secrets.")
else:
    genai.configure(api_key=api_key)


# -------------------------
#  HEADER SECTION (React-style)
# -------------------------
st.markdown("""
<div style="
    text-align:center; 
    padding-top:40px; 
    padding-bottom:20px;
">
    <div style="
        display:inline-flex; 
        align-items:center; 
        padding:12px 20px; 
        background:#ffffff; 
        border-radius:18px; 
        box-shadow:0px 4px 12px rgba(0,0,0,0.05); 
        border:1px solid #d1fae5;
        margin-bottom:20px;
    ">
        <span style="background:#d1fae5; padding:10px; border-radius:12px; margin-right:12px; color:#047857;">
            🎙
        </span>
        <span style="font-size:22px; font-weight:700; color:#0f172a;">UrduScribe AI</span>
    </div>
    <h1 style="font-size:42px; font-weight:900; color:#0f172a; margin:0;">
        Audio to <span style="background: -webkit-linear-gradient(#059669, #10b981); -webkit-background-clip:text; color:transparent;">Urdu Text</span>
    </h1>
    <p style="color:#6b7280; font-size:18px; max-width:650px; margin:auto;">
        Upload an MP3/WAV or record your voice. Uses Gemini 2.5 Flash for accurate, fast Urdu transcription.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------
#  AUDIO INPUT
# -------------------------
uploaded_file = st.file_uploader("Upload audio (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

transcribe_clicked = st.button("Transcribe", use_container_width=True, type="primary")

if transcribe_clicked:
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        with st.spinner("Processing audio… ⏳"):
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, uploaded_file.name)

            # Save file
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # Send to Gemini
            model = genai.GenerativeModel("models/gemini-2.0-flash")

            audio_bytes = Path(temp_audio_path).read_bytes()

            result = model.generate_content([
                genai.types.Part.from_bytes(audio_bytes, mime_type=uploaded_file.type),
                "Transcribe this audio to Urdu text."
            ])

            text_output = result.text

        st.success("Transcription completed ✔")

        st.markdown("### 📝 Urdu Transcription")
        st.markdown(f"<div class='urdu-text'>{text_output}</div>", unsafe_allow_html=True)

        st.download_button(
            "Download Transcript",
            data=text_output,
            file_name="urdu_transcript.txt",
            mime="text/plain"
        )
