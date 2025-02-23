from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load the model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

def whisper_transcription(audio_file_path):
    # Load and resample audio
    speech, _ = librosa.load(audio_file_path, sr=16000)
    # Process the speech audio
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # Decode the transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)
    return transcription[0]

import streamlit as st
from your_whisper_script import whisper_transcription

st.title('Speech to Text Conversion using Whisper Large v2')
audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])

if audio_file is not None:
    # Save the file locally for processing
    audio_file_path = audio_file.name
    with open(audio_file_path, 'wb') as f:
        f.write(audio_file.getbuffer())
    # Perform transcription
    transcription = whisper_transcription(audio_file_path)
    st.write("Transcription:")
    st.text_area("Result", value=transcription, height=150, max_chars=None)


