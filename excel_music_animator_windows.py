import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import simpleaudio as sa
import time
import os
import wave

# ==============================
# 🎵 NOTE → FREQUENCY HELPER
# ==============================
# Maps note names to frequencies (Hz)
note_freqs = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
    'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
}

# ==============================
# 🎹 INSTRUMENT SAMPLE LOADER
# ==============================
def load_wav(filename):
    """Load a WAV file and return audio array and sample rate"""
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_samples = wf.getnframes()
        audio = wf.readframes(n_samples)
        audio = np.frombuffer(audio, dtype=np.int16)
    return audio.astype(np.float32), sample_rate

# Add your WAV files here (place them in the same folder as this script)
instruments = {
    "piano": "piano_C4.wav",
    "guitar": "guitar_C4.wav",
    "flute": "flute_C4.wav"
}

current_instrument = "piano"  # choose default instrument

def synthesize(note, duration, velocity=100):
    """Generate a tone either from a sample WAV or fallback sinewave"""
    if note not in note_freqs:
        return np.zeros(int(44100 * duration), dtype=np.float32)

    freq = note_freqs[note]
    sample_rate = 44100

    if os.path.exists(instruments[current_instrument]):
        # Load base sample (currently ignoring pitch shifting for simplicity)
        base_audio, base_sr = load_wav(instruments[current_instrument])
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
    else:
        # Fallback sine wave if WAV not found
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)

    audio = tone * (velocity / 127.0)  # adjust volume based on velocity
    return audio.astype(np.float32)

# ==============================
# 🎼 MAIN PLAYER WITH ANIMATION
# ==============================
def play_score(df):
    """Play notes with animation bars from Excel score"""
    fig, ax = plt.subplots()
    bars = ax.barh(range(len(df)), df['duration'], left=df['start'],
                   color=df['color'], align='center')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['note'])
    ax.set_xlabel("Time (s)")
    ax.set_title("Excel Music Animator")

    start_time = time.time()

    def update(frame):
        current_time = time.time() - start_time
        for i, row in df.iterrows():
            # Play note when start time is reached
            if abs(current_time - row['start']) < 0.05:
                audio = synthesize(row['note'], row['duration'], row['velocity'])
                sa.play_buffer((audio * 32767).astype(np.int16), 1, 2, 44100)
        return bars

    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Load Excel score
    df = pd.read_excel("score.xlsx")

    # Ensure your Excel has these columns: note, start, duration, velocity, color
    # Example:
    # note    start   duration    velocity    color
    # C4      0.0     0.5         100       red
    # E4      0.5     0.5         100       blue

    # Play the music with animation
    play_score(df)
