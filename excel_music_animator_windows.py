"""
excel_music_animator_windows.py
Windows-ready: reads score.xlsx, synthesizes audio, animates, auto-reloads on file save,
and exports MIDI.

Place score.xlsx (Sheet1) in same folder. Columns: start, duration, note, velocity, color

Usage:
  python excel_music_animator_windows.py
Options:
  --no-watch   run once and exit (no auto-reload)
  --no-audio   skip audio playback (useful for debugging)
"""

import argparse
import time
import threading
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import simpleaudio as sa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from mido import Message, MidiFile, MidiTrack
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------- Configuration ----------
SAMPLE_RATE = 44100
MASTER_VOLUME = 0.7
NOTE_RELEASE = 0.02
SCORE_FILE = Path("score.xlsx")
AUTO_RELOAD = True
# -----------------------------------

# robust note name map
NOTE_BASE = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
    'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
}

def note_to_midi(note_str: str) -> int:
    s = note_str.strip().upper().replace('♯', '#').replace('♭', 'B')
    # find split between name and octave (octave digits at end, allow negatives)
    i = len(s) - 1
    while i >= 0 and (s[i].isdigit() or s[i] == '-'):
        i -= 1
    note_name = s[:i+1]
    octave_str = s[i+1:] if (i+1) < len(s) else "4"
    octave = int(octave_str)
    # fix common synonyms
    if note_name not in NOTE_BASE:
        # try replace '♭' already done; try swap 'B' meaning Bb etc.
        raise ValueError(f"Unknown note name: {note_name}")
    semitone_index = NOTE_BASE[note_name]
    midi_number = 12 + semitone_index + (octave * 12)
    return midi_number

def midi_to_freq(midi_num: int) -> float:
    return 440.0 * (2 ** ((midi_num - 69) / 12.0))

def note_to_freq(note_str: str) -> float:
    return midi_to_freq(note_to_midi(note_str))

def synth_note(freq, duration, velocity=100, sample_rate=SAMPLE_RATE):
    # additive-ish simple synth (carrier + 2 harmonics) with ADSR-like envelope
    n_samples = max(1, int(sample_rate * duration))
    t = np.linspace(0, duration, n_samples, endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    harmonic = 0.5 * np.sin(2 * np.pi * 2 * freq * t) + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    raw = carrier + harmonic
    attack = min(0.02, duration * 0.2)
    release = min(0.12, duration * 0.25 + NOTE_RELEASE)
    env = np.ones_like(t)
    atk_samp = int(sample_rate * attack)
    if atk_samp > 0:
        env[:atk_samp] = np.linspace(0, 1, atk_samp)
    rel_samp = int(sample_rate * release)
    if rel_samp > 0:
        env[-rel_samp:] *= np.linspace(1, 0, rel_samp)
    amplitude = (velocity / 127.0) * MASTER_VOLUME
    out = raw * env * amplitude
    # gentle normalization
    maxv = np.max(np.abs(out))
    if maxv > 0.999:
        out = out / maxv * 0.999
    return out.astype(np.float32)

def load_score(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_excel(path, sheet_name=0)
    # minimal validation & defaults
    for col in ['start', 'duration', 'note']:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in score.xlsx")
    if 'velocity' not in df.columns:
        df['velocity'] = 100
    if 'color' not in df.columns:
        df['color'] = '#4da6ff'
    df = df.fillna({'velocity': 100, 'color': '#4da6ff'})
    events = []
    for idx, row in df.iterrows():
        start = float(row['start'])
        dur = float(row['duration'])
        note = str(row['note'])
        vel = int(row['velocity'])
        color = str(row['color'])
        try:
            freq = note_to_freq(note)
        except Exception as ex:
            raise ValueError(f"Error parsing note '{note}' at row {idx+2}: {ex}")
        events.append({
            'start': start,
            'duration': dur,
            'note': note,
            'freq': freq,
            'velocity': vel,
            'color': color,
            'idx': int(idx)
        })
    return events

def synth_mix(events):
    total_dur = max(e['start'] + e['duration'] for e in events) + 0.05
    mix = np.zeros(int(SAMPLE_RATE * total_dur), dtype=np.float32)
    for e in events:
        w = synth_note(e['freq'], e['duration'], velocity=e['velocity'])
        s = int(e['start'] * SAMPLE_RATE)
        mix[s:s+len(w)] += w
    peak = np.max(np.abs(mix))
    if peak > 0.999:
        mix = mix / peak * 0.999
    pcm16 = (mix * 32767).astype(np.int16)
    return pcm16, total_dur

def export_midi(events, filename="score_export.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    # We'll add notes as absolute ticks with 480 ticks per beat,
    # use tempo = 500000 (120 BPM) so times convert roughly but midi here is simple
    ticks_per_second = 480 / 0.5  # approximate if 120 bpm -> 0.5s per beat -> 960 ticks/sec; but it's fine for DAWs
    last_tick = 0
    for e in sorted(events, key=lambda x: x['start']):
        on_tick = int(e['start'] * ticks_per_second)
        off_tick = int((e['start'] + e['duration']) * ticks_per_second)
        delta_on = max(0, on_tick - last_tick)
        midi_note = note_to_midi(e['note'])
        track.append(Message('note_on', note=midi_note, velocity=e['velocity'], time=delta_on))
        track.append(Message('note_off', note=midi_note, velocity=0, time=off_tick - on_tick))
        last_tick = off_tick
    mid.save(filename)
    print(f"[MIDI] Saved {filename}")

class ScoreWatcher(FileSystemEventHandler):
    def __init__(self, path, callback):
        super().__init__()
        self._path = str(path.resolve())
        self._callback = callback

    def on_modified(self, event):
        # only react to our file changes
        if event.src_path.endswith(self._path) or event.src_path.endswith(".xlsx"):
            print("[watcher] Detected change in score file. Reloading...")
            try:
                self._callback()
            except Exception as ex:
                print("Reload failed:", ex)

def run_player(events, play_audio=True):
    # Synthesize mix
    pcm16, total_dur = synth_mix(events)
    audio_bytes = pcm16.tobytes()
    play_obj = None
    if play_audio:
        try:
            play_obj = sa.play_buffer(audio_bytes, 1, 2, SAMPLE_RATE)
        except Exception as ex:
            print("Audio playback failed:", ex)
            play_obj = None

    # Setup visualization
    events_sorted = sorted(events, key=lambda x: (x['start'], x['idx']))
    fig, ax = plt.subplots(figsize=(10, max(3, len(events_sorted)*0.4)))
    ax.set_xlim(0, total_dur)
    ax.set_ylim(0, len(events_sorted))
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Excel → Musical Animation (close window to stop)")

    bars = []
    for i, e in enumerate(events_sorted):
        rect = Rectangle((e['start'], i + 0.1), e['duration'], 0.8, facecolor=e['color'], alpha=0.35)
        ax.add_patch(rect)
        bars.append(rect)
        ax.text(e['start'] + 0.02, i + 0.5, f"{e['note']} ({e['velocity']})", va='center', fontsize=8)

    play_head_line, = ax.plot([0, 0], [0, len(events_sorted)], color='k', linewidth=1)

    def active_indices_at(t):
        active = []
        for i, e in enumerate(events_sorted):
            if e['start'] <= t < (e['start'] + e['duration']):
                active.append(i)
        return active

    FPS = 30
    frames = int(total_dur * FPS) + 2

    def update(frame):
        t = frame / FPS
        play_head_line.set_xdata([t, t])
        active = set(active_indices_at(t))
        for i, rect in enumerate(bars):
            if i in active:
                rect.set_alpha(1.0)
                rect.set_edgecolor('black')
                rect.set_linewidth(1.2)
            else:
                rect.set_alpha(0.25)
                rect.set_edgecolor(None)
        return bars + [play_head_line]

    ani = animation.FuncAnimation(fig, func=update, frames=frames, interval=1000/FPS, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

    if play_obj:
        play_obj.wait_done()

def load_and_play(play_audio=True, export_midi_file=True):
    try:
        events = load_score(SCORE_FILE)
    except Exception as ex:
        print("Error loading score:", ex)
        return
    print(f"[INFO] Loaded {len(events)} events. Synthesizing and playing...")
    if export_midi_file:
        try:
            export_midi(events, filename="score_export.mid")
        except Exception as ex:
            print("MIDI export failed:", ex)
    run_player(events, play_audio=play_audio)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-watch", action="store_true", help="Run once and exit (no auto-reload)")
    parser.add_argument("--no-audio", action="store_true", help="Do not play audio (visual only)")
    args = parser.parse_args()

    play_audio = not args.no_audio
    watch_mode = AUTO_RELOAD and not args.no_watch

    # initial run
    load_and_play(play_audio=play_audio, export_midi_file=True)

    if watch_mode:
        # Setup watchdog to re-run when file changes
        def on_change():
            # when watch triggers, re-run load_and_play in a separate thread so GUI loop can close
            t = threading.Thread(target=load_and_play, kwargs={"play_audio":play_audio, "export_midi_file":True}, daemon=True)
            t.start()

        event_handler = ScoreWatcher(SCORE_FILE, on_change)
        observer = Observer()
        observer.schedule(event_handler, path=".", recursive=False)
        observer.start()
        print("[watcher] Watching for changes to score.xlsx. Edit & save to re-run. Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
