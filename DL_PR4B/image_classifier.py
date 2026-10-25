"""
# Task B1: Hand Gesture Classifier & Gesture Volume Controller
# -------------------------------------------------------------
# Application Description:
# This application detects hand gestures in real-time from a webcam stream
# using a Teachable Machine Image Classifier model.
#
# Recognized Gestures:
# 1. Thumbs Up   -> Automatically increases system audio volume.
# 2. Thumbs Down -> Automatically decreases system audio volume.
# 3. Peace Sign  -> Informative gesture overlay.
# 4. Open Palm   -> Informative gesture overlay / Standby.
#
# How to use:
# - Run the script. A webcam window titled 'Image Classifier - Gesture Control' will appear.
# - Show a 'Thumbs Up' gesture to trigger 'Volume Up'.
# - Show a 'Thumbs Down' gesture to trigger 'Volume Down'.
# - A confidence bar and detected class name are overlaid on the video frame.
# - Press 'q' on your keyboard to close the application and save a screenshot to output/b1_image_result.jpg.
"""

import os
import sys
import time
import numpy as np
import cv2
import pyautogui

# Try loading from local module loader
try:
    from model_loader import load_tm_model, RobustVideoCapture
except ImportError:
    from DL_PR4B.model_loader import load_tm_model, RobustVideoCapture


def main():
    os.makedirs('DL_PR4B/output', exist_ok=True)
    model_path = os.path.join('DL_PR4B', 'models', 'image', 'keras_model.h5')
    labels_path = os.path.join('DL_PR4B', 'models', 'image', 'labels.txt')

    print("[B1] Loading image classifier model...")
    model, labels = load_tm_model(model_path, labels_path)
    print(f"[B1] Loaded classes: {labels}")

    # Open webcam with robust fallback
    cap = RobustVideoCapture(0)
    print("[B1] Webcam active. Press 'q' to exit.")

    last_volume_action = time.time()
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # (d) Preprocess: resize to 224x224, normalize to [-1, 1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb_frame, (224, 224))
        img = np.asarray(img, dtype=np.float32)
        img = (img / 127.5) - 1.0
        img = np.expand_dims(img, axis=0)

        # (e) Predict
        prediction = model.predict(img, verbose=0)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])
        class_name = labels[class_idx] if class_idx < len(labels) else f"Class {class_idx}"

        # (f) B1.5 Build something useful: Gesture Volume Control
        # Trigger volume adjustment with a small cooldown to avoid rapid firing
        now = time.time()
        if confidence > 0.75 and (now - last_volume_action > 0.4):
            lower_name = class_name.lower()
            if "up" in lower_name:
                try:
                    pyautogui.press('volumeup')
                    last_volume_action = now
                except Exception:
                    pass
            elif "down" in lower_name:
                try:
                    pyautogui.press('volumedown')
                    last_volume_action = now
                except Exception:
                    pass

        # Overlay class name and confidence bar on frame
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (380, 80), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (380, 80), (100, 200, 100), 2)

        # Text label
        label_text = f"{class_name}: {confidence * 100:.1f}%"
        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Confidence bar
        bar_width = int(340 * confidence)
        cv2.rectangle(frame, (20, 55), (360, 70), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 55), (20 + bar_width, 70), (0, 220, 120), -1)

        last_frame = frame.copy()
        cv2.imshow("Image Classifier - Gesture Control", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save screenshot to DL_PR4B/output/b1_image_result.jpg
    if last_frame is not None:
        out_path = os.path.join('DL_PR4B', 'output', 'b1_image_result.jpg')
        cv2.imwrite(out_path, last_frame)
        print(f"[B1] Screenshot saved to {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
