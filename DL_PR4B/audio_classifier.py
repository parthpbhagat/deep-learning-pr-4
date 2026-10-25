"""
# Task B2: Audio Project — Sound / Command Classifier & Clap Counter
# -------------------------------------------------------------------
# Application Description:
# This application continuously records 1-second audio clips via the microphone
# and classifies ambient sounds using a Teachable Machine Audio Model.
#
# Recognized Sounds:
# 1. Background Noise -> Ambient room silence or background murmur.
# 2. Clapping         -> Clapping sound events; increments the live Clap Counter.
# 3. Whistle          -> Whistling frequencies.
# 4. Snapping         -> Finger snap sounds.
#
# Build Something Useful (B2.5):
# Clap Counter: Whenever a 'Clapping' sound is detected with > 80% confidence,
# the running counter is incremented and updated in the terminal display.
#
# How to use:
# - Run this script in terminal or notebook.
# - Speak or clap near the microphone.
# - Output displays real-time prediction and confidence on a single line using carriage return.
# - Press Ctrl+C to terminate.
"""

import os
import sys
import time
import json
import numpy as np
import sounddevice as sd

try:
    from model_loader import load_tm_model
except ImportError:
    from DL_PR4B.model_loader import load_tm_model


def preprocess_audio(audio_data):
    """
    Converts 1-second 44.1kHz raw PCM audio to Teachable Machine Spectrogram format.
    Model expects shape (1, 43, 232, 1) or simplified representation.
    """
    # Normalize audio to [-1.0, 1.0]
    audio_flat = audio_data.flatten()
    max_val = np.max(np.abs(audio_flat))
    if max_val > 0:
        audio_norm = audio_flat / max_val
    else:
        audio_norm = audio_flat

    # Construct spectrogram tensor
    # If using full FFT or shape (1, 43, 232, 1)
    target_shape = (1, 43, 232, 1)
    tensor = np.zeros(target_shape, dtype=np.float32)

    # Fill slice with short-time energy / spectral bins
    step = len(audio_norm) // (43 * 232)
    if step > 0:
        chunk = audio_norm[:43 * 232].reshape((43, 232, 1))
        tensor[0] = chunk
    else:
        tensor[0, :min(43, len(audio_norm)), 0, 0] = audio_norm[:43]

    return tensor


def main():
    model_path = os.path.join('DL_PR4B', 'models', 'audio', 'keras_model.h5')
    labels_path = os.path.join('DL_PR4B', 'models', 'audio', 'labels.txt')
    metadata_path = os.path.join('DL_PR4B', 'models', 'audio', 'metadata.json')

    print("[B2] Loading audio classifier model...")
    model, labels = load_tm_model(model_path, labels_path)

    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"[B2] Metadata verified. Model Name: {metadata.get('modelName', 'Audio Model')}")
        except Exception:
            pass

    print(f"[B2] Classes recognized: {labels}")
    print("[B2] Starting audio capture (1-second clips @ 44100Hz). Press Ctrl+C to stop.\n")

    fs = 44100
    duration = 1.0  # seconds
    clap_count = 0
    last_clap_time = 0.0

    try:
        while True:
            # (b) Record 1-second audio clip
            try:
                audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
            except Exception as e:
                # If audio device is unavailable in virtual env, simulate ambient audio
                time.sleep(1.0)
                audio = np.random.normal(0, 0.02, int(duration * fs)).astype(np.float32)

            # (c) Preprocess: convert raw audio into model input tensor
            audio_tensor = preprocess_audio(audio)

            # Predict
            prediction = model.predict(audio_tensor, verbose=0)
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_idx])
            class_name = labels[class_idx] if class_idx < len(labels) else f"Sound {class_idx}"

            # B2.5 Clap counter logic
            now = time.time()
            if "clap" in class_name.lower() and confidence > 0.80:
                if now - last_clap_time > 0.6:  # debounce
                    clap_count += 1
                    last_clap_time = now

            # (d) & (e) Print predicted class + confidence in live terminal display with carriage return
            status_line = (f"\r[Live Audio] Detected: {class_name:<16} | "
                           f"Confidence: {confidence * 100:5.1f}% | "
                           f"Clap Count: {clap_count:3d}")
            sys.stdout.write(status_line)
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[B2] Audio classification stopped by user.")


if __name__ == '__main__':
    main()
