import torch
import numpy as np
from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
import pyaudio
from gtts import gTTS  # Google Text-to-Speech for the woman's voice
import io
import soundfile as sf

# Load the AudioLDM2 pipeline
pipeline = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2-music", torch_dtype=torch.float16
)
pipeline.to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)
pipeline.enable_model_cpu_offload()

# Initialize PyAudio for playing sound
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)

# Function to play audio
def play_audio(audio_data, rate=16000):
    stream.write(audio_data.astype(np.float32))

# Function to convert text to speech (woman's voice introduction)
def generate_intro(user_name):
    intro_text = f"DJ {user_name}, we the best music"
    tts = gTTS(text=intro_text, lang='en', slow=False)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    intro_audio, sr = sf.read(audio_fp, dtype='float32')  # Ensure float32 format for correct playback
    return intro_audio

# Get user input for name and age
user_name = input("Please enter your name: ")
user_age = int(input("Please enter your age: "))

# Create a prompt for music generation based on the user's name and age
song_prompt = f"A {user_age}-year-old music with the name meaning similar to {user_name}"

# Inform the user about the generated music
print(f"Generating a {user_age}-year-old song with a name meaning similar to '{user_name}'...")

# Generate music using the AudioLDM2 pipeline
music_segment = pipeline(song_prompt, num_inference_steps=200, audio_length_in_s=45).audios[0]

# Generate the DJ introduction
intro_audio = generate_intro(user_name)

# Concatenate the DJ intro with the music
combined_audio = np.concatenate((intro_audio, music_segment))

# Play the combined audio
print(f"Now playing: '{song_prompt}' featuring the intro 'DJ {user_name}, we the best music'")
play_audio(combined_audio)

# Close the stream after playback
stream.stop_stream()
stream.close()
p.terminate()