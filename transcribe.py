import whisper
import torch
from pyannote.audio import Pipeline
import requests
from urllib.parse import urlparse
from pathlib import Path
import os
from pydub import AudioSegment
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio_with_speakers(audio_url):
    """
    Transcribe audio from URL using Whisper and perform speaker diarization
    
    Parameters:
        audio_url (str): URL to the MP3 file
        
    Returns:
        str: Transcribed text with speaker labels
    """
    try:
        # Download the audio file if it's a URL
        if urlparse(audio_url).scheme in ('http', 'https'):
            response = requests.get(audio_url)
            temp_path = "temp_audio.mp3"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
        else:
            temp_path = audio_url

        # Load audio using pydub (handles various formats)
        audio = AudioSegment.from_file(temp_path)
        audio.export("temp_processed.wav", format="wav")

        # Initialize Whisper model
        model = whisper.load_model("base")
        
        # Transcribe using Whisper
        result = model.transcribe("temp_processed.wav")
        
        print(f"Transcribed using Whisper: {result['text'][:100]}")
        print(f"Hugging Face Token: {os.getenv('HUGGING_FACE_TOKEN')}")
        # Initialize speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
        )
        
        # Perform diarization
        diarization = pipeline("temp_processed.wav", num_speakers=2)
        
        # Combine transcription with speaker labels
        segments = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            
            # Find corresponding text from whisper transcription
            segment_text = ""
            for whisper_segment in result["segments"]:
                w_start = whisper_segment["start"]
                w_end = whisper_segment["end"]
                
                # Check for overlap
                if (w_start >= start_time and w_start < end_time) or \
                   (w_end > start_time and w_end <= end_time):
                    segment_text += " " + whisper_segment["text"]
            
            if segment_text.strip():
                segments.append(f"{speaker}: {segment_text.strip()}")
        
        # Clean up temporary files
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
        if os.path.exists("temp_processed.wav"):
            os.remove("temp_processed.wav")
            
        return "\n".join(segments)
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"