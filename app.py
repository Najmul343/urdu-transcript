import streamlit as st
from faster_whisper import WhisperModel
import os
from pydub import AudioSegment
import torch
import time

# Page Config
st.set_page_config(page_title="Urdu Transcriber (Live)", page_icon="🎤")

st.title("🎤 Urdu Audio Transcriber")
st.write("Upload an audio/video file. (Live streaming output)")

# Hardcoded Model
model_size = "small"

# File Uploader
uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Transcribe Now"):
        
        # Create a placeholder for live text updates
        status_text = st.empty()
        live_text_box = st.empty()
        
        try:
            status_text.text("Processing audio...")
            
            # --- Save & Convert ---
            temp_filename = "temp_upload" + os.path.splitext(uploaded_file.name)[1]
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status_text.text("Converting to WAV...")
            audio = AudioSegment.from_file(temp_filename)
            audio = audio.set_frame_rate(16000).set_channels(1)
            wav_path = "converted_audio.wav"
            audio.export(wav_path, format="wav")

            # --- Load Model ---
            status_text.text(f"Loading {model_size} model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            model = WhisperModel(model_size, device=device, compute_type=compute_type)

            # --- Transcribe with Streaming ---
            status_text.text("Transcribing... (Watch below)")
            
            segments, info = model.transcribe(wav_path, language="ur", beam_size=5)
            
            full_text = ""
            
            # --- THIS IS THE MAGIC LOOP ---
            for segment in segments:
                new_line = segment.text.strip()
                full_text += new_line + " "
                
                # Update the text box immediately!
                live_text_box.markdown(f"### 📝 Live Transcript:\n\n{full_text}")
                
                # Optional: smooth scroll feel
                time.sleep(0.01) 

            status_text.success(f"Finished! (Confidence: {info.language_probability:.0%})")
            
            # Cleanup
            if os.path.exists(temp_filename): os.remove(temp_filename)
            if os.path.exists(wav_path): os.remove(wav_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")





'''
import streamlit as st
from faster_whisper import WhisperModel
import os
from pydub import AudioSegment
import torch

# Page Config
st.set_page_config(page_title="Urdu Transcriber (Tiny)", page_icon="🎤")

st.title("🎤 Urdu Audio Transcriber")
st.write("Upload an audio/video file. (Using **Tiny** model for speed)")

# Hardcoded Model
model_size = "tiny"

# File Uploader
uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file) # Play the audio/video to verify
    
    if st.button("Transcribe Now"):
        with st.spinner(f"Processing with {model_size} model..."):
            try:
                # --- Save uploaded file temporarily ---
                temp_filename = "temp_upload" + os.path.splitext(uploaded_file.name)[1]
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # --- Step 1: Convert to WAV ---
                st.text("Converting audio format...")
                audio = AudioSegment.from_file(temp_filename)
                audio = audio.set_frame_rate(16000).set_channels(1)
                wav_path = "converted_audio.wav"
                audio.export(wav_path, format="wav")

                # --- Step 2: Load Model ---
                # Detect hardware (Force int8 on CPU to prevent errors)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                # Initialize Model
                model = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type=compute_type
                )

                # --- Step 3: Transcribe ---
                st.text("Transcribing in Urdu...")
                segments, info = model.transcribe(
                    wav_path, 
                    language="ur", 
                    beam_size=5
                )

                # --- Step 4: Output ---
                st.success(f"Done! (Confidence: {info.language_probability:.0%})")
                
                full_text = ""
                output_container = st.empty()
                segments_list = list(segments) 
                
                for segment in segments_list:
                    text = segment.text.strip()
                    full_text += text + " "

                # Display Final Text
                st.markdown("### 📝 Urdu Transcription:")
                st.text_area("Result:", full_text, height=300)
                
                # Cleanup temp files
                if os.path.exists(temp_filename): os.remove(temp_filename)
                if os.path.exists(wav_path): os.remove(wav_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
'''
