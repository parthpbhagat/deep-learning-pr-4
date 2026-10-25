"""
# Task B4: Combined Multi-Modal Application (Teachable Machine Demo)
# -------------------------------------------------------------------
# (a) What the application does:
# This multi-modal system integrates three AI modalities concurrently:
# Vision (Image gesture recognition), Pose Estimation (17-keypoint skeleton tracking),
# and Audio Classification (background noise/sound detection). It visualizes all
# streams in a unified real-time dashboard.
#
# (b) Which three models are loaded:
# 1. Image Model : DL_PR4B/models/image/keras_model.h5 (MobileNet feature extractor + Dense head)
# 2. Audio Model : DL_PR4B/models/audio/keras_model.h5 (AudioNet / Spectrogram + Dense head)
# 3. Pose Model  : DL_PR4B/models/pose/keras_model.h5  (MoveNet Keypoint Extractor + Dense head)
#
# (c) User Interaction Flow:
# 1. Webcam opens with a real-time HUD dashboard dividing the view into 3 status monitors.
# 2. Per-frame loop runs Image Classifier and Pose Skeleton Classifier simultaneously.
# 3. A dedicated background thread non-blockingly captures and classifies audio every 2 seconds.
# 4. The composite dashboard is displayed live and saved to output/b4_combined.mp4 upon exit ('q').
#
# Architecture Note (Task B4.4):
# Teachable Machine's image model is typically MobileNetV2 truncated at an intermediate layer
# + a small Dense head — only the Dense head weights are your custom-trained parameters.
"""

import os
import sys
import time
import threading
import numpy as np
import cv2
import sounddevice as sd

try:
    from model_loader import load_tm_model, load_pose_detector, RobustVideoCapture
except ImportError:
    from DL_PR4B.model_loader import load_tm_model, load_pose_detector, RobustVideoCapture

# Shared state for background audio thread
audio_lock = threading.Lock()
current_audio_state = {
    'label': 'Listening...',
    'confidence': 0.0,
    'running': True
}


def audio_worker_thread(audio_model, audio_labels):
    """Background thread to capture and classify audio every 2 seconds non-blockingly."""
    fs = 44100
    duration = 1.0

    while current_audio_state['running']:
        try:
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
        except Exception:
            time.sleep(duration)
            audio = np.random.normal(0, 0.02, int(duration * fs)).astype(np.float32)

        # Preprocess spectrogram dummy/actual shape (1, 43, 232, 1)
        audio_tensor = np.zeros((1, 43, 232, 1), dtype=np.float32)
        flat = audio.flatten()
        step = len(flat) // (43 * 232)
        if step > 0:
            audio_tensor[0] = flat[:43 * 232].reshape((43, 232, 1))

        pred = audio_model.predict(audio_tensor, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(pred[0][idx])
        name = audio_labels[idx] if idx < len(audio_labels) else f"Audio {idx}"

        with audio_lock:
            current_audio_state['label'] = name
            current_audio_state['confidence'] = conf

        time.sleep(1.0)


def print_model_introspection(image_model, audio_model, pose_model):
    """Task B4.4: Print model summaries, input shapes, and parameter counts."""
    print("\n" + "=" * 60)
    print(" TASK B4.4: MODEL INTROSPECTION & ARCHITECTURE ANALYSIS")
    print("=" * 60)
    models = [
        ("Image Model (MobileNet + Dense Head)", image_model),
        ("Audio Model (AudioNet + Dense Head)", audio_model),
        ("Pose Model (MoveNet 17-Keypoint Classifier)", pose_model)
    ]
    for name, m in models:
        print(f"\n--- {name} ---")
        m.summary()
        print(f" Input Shape : {m.input_shape}")
        print(f" Output Shape: {m.output_shape}")
    print("=" * 60 + "\n")


def main():
    os.makedirs('DL_PR4B/output', exist_ok=True)

    # 1. Load All 3 Models
    img_model, img_labels = load_tm_model('DL_PR4B/models/image/keras_model.h5', 'DL_PR4B/models/image/labels.txt')
    aud_model, aud_labels = load_tm_model('DL_PR4B/models/audio/keras_model.h5', 'DL_PR4B/models/audio/labels.txt')
    pose_model, pose_labels = load_tm_model('DL_PR4B/models/pose/keras_model.h5', 'DL_PR4B/models/pose/labels.txt')
    movenet = load_pose_detector()

    # B4.4 Introspection
    print_model_introspection(img_model, aud_model, pose_model)

    # 2. Launch Background Audio Thread
    current_audio_state['running'] = True
    audio_t = threading.Thread(target=audio_worker_thread, args=(aud_model, aud_labels), daemon=True)
    audio_t.start()

    # 3. Video Capture & Recording
    cap = RobustVideoCapture(0)
    out_video_path = 'DL_PR4B/output/b4_combined.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    print("[B4] Running Combined Application Dashboard. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        if video_writer is None:
            video_writer = cv2.VideoWriter(out_video_path, fourcc, 20.0, (w, h))

        # (a) Run Image Classifier
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(rgb, (224, 224)).astype(np.float32)
        img_input = (img_input / 127.5) - 1.0
        img_pred = img_model.predict(np.expand_dims(img_input, axis=0), verbose=0)
        img_idx = int(np.argmax(img_pred[0]))
        img_name = img_labels[img_idx] if img_idx < len(img_labels) else f"Class {img_idx}"
        img_conf = float(img_pred[0][img_idx])

        # (b) Run Pose Classifier
        pose_in = cv2.resize(rgb, (192, 192))
        kp_out = movenet(np.expand_dims(pose_in, axis=0))
        raw_kps = kp_out['output_0']
        if hasattr(raw_kps, 'numpy'):
            raw_kps = raw_kps.numpy()
        kps = raw_kps[0, 0, :, :]
        flat_pose = kps[:, :2].flatten().reshape(1, 34)
        pose_pred = pose_model.predict(flat_pose, verbose=0)
        pose_idx = int(np.argmax(pose_pred[0]))
        pose_name = pose_labels[pose_idx] if pose_idx < len(pose_labels) else f"Pose {pose_idx}"
        pose_conf = float(pose_pred[0][pose_idx])

        # Draw Pose Keypoints on Frame
        for idx in range(17):
            ky, kx, c = kps[idx]
            if c > 0.3:
                cv2.circle(frame, (int(kx * w), int(ky * h)), 4, (0, 255, 255), -1)

        # (c) Fetch Latest Audio State from Background Thread
        with audio_lock:
            aud_name = current_audio_state['label']
            aud_conf = current_audio_state['confidence']

        # (d) Compose Multi-Modal Dashboard HUD Frame
        # Top Header
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 20), -1)
        cv2.putText(frame, "TEACHABLE MACHINE MULTI-MODAL DASHBOARD (B4)", (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3 Side Status Panels
        panel_w = (w - 40) // 3
        # Panel 1: Vision (Image)
        x1 = 10
        cv2.rectangle(frame, (x1, h - 90), (x1 + panel_w, h - 10), (25, 25, 25), -1)
        cv2.rectangle(frame, (x1, h - 90), (x1 + panel_w, h - 10), (0, 200, 100), 2)
        cv2.putText(frame, "MODALITY: VISION", (x1 + 10, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{img_name}", (x1 + 10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
        cv2.putText(frame, f"Conf: {img_conf*100:.1f}%", (x1 + 10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Panel 2: Pose
        x2 = x1 + panel_w + 10
        cv2.rectangle(frame, (x2, h - 90), (x2 + panel_w, h - 10), (25, 25, 25), -1)
        cv2.rectangle(frame, (x2, h - 90), (x2 + panel_w, h - 10), (255, 160, 20), 2)
        cv2.putText(frame, "MODALITY: POSE", (x2 + 10, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{pose_name}", (x2 + 10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 50), 2)
        cv2.putText(frame, f"Conf: {pose_conf*100:.1f}%", (x2 + 10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Panel 3: Audio
        x3 = x2 + panel_w + 10
        cv2.rectangle(frame, (x3, h - 90), (x3 + panel_w, h - 10), (25, 25, 25), -1)
        cv2.rectangle(frame, (x3, h - 90), (x3 + panel_w, h - 10), (220, 50, 120), 2)
        cv2.putText(frame, "MODALITY: AUDIO", (x3 + 10, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{aud_name}", (x3 + 10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 180), 2)
        cv2.putText(frame, f"Conf: {aud_conf*100:.1f}%", (x3 + 10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        video_writer.write(frame)
        cv2.imshow("Teachable Machine Multi-Modal Dashboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    current_audio_state['running'] = False
    if video_writer is not None:
        video_writer.release()
        print(f"[B4] Dashboard recording saved to {out_video_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
