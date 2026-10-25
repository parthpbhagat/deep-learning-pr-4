"""
# Task B3: Body Pose Classifier & Exercise Rep Counter
# ------------------------------------------------------
# Application Description:
# This application captures webcam frames, extracts 17 body keypoints using MoveNet,
# and classifies the body pose using an exported Teachable Machine Pose model.
#
# Recognized Poses:
# 1. Squat Down
# 2. Squat Up
# 3. Rest
#
# Build Something Useful (B3.5) — Exercise Rep Counter:
# The application tracks squat repetitions by monitoring state transitions from
# 'Squat Down' to 'Squat Up'. When a user completes the upward motion,
# the squat counter is incremented and displayed directly on the screen.
#
# Visuals:
# - Draws 17 keypoints and connecting skeleton lines on the user's body.
# - Displays current pose name, confidence score, and completed repetition count.
# - Saves output video to output/b3_pose_result.mp4 upon quitting ('q').
"""

import os
import sys
import time
import numpy as np
import cv2

try:
    from model_loader import load_tm_model, load_pose_detector, RobustVideoCapture
except ImportError:
    from DL_PR4B.model_loader import load_tm_model, load_pose_detector, RobustVideoCapture

# 17 Keypoints connections for drawing skeleton
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Facial connections
    (5, 6),                                # Shoulders
    (5, 7), (7, 9),                        # Left arm
    (6, 8), (8, 10),                       # Right arm
    (5, 11), (6, 12), (11, 12),            # Torso
    (11, 13), (13, 15),                    # Left leg
    (12, 14), (14, 16)                     # Right leg
]


def main():
    os.makedirs('DL_PR4B/output', exist_ok=True)
    model_path = os.path.join('DL_PR4B', 'models', 'pose', 'keras_model.h5')
    labels_path = os.path.join('DL_PR4B', 'models', 'pose', 'labels.txt')

    print("[B3] Loading Teachable Machine pose classifier...")
    pose_model, labels = load_tm_model(model_path, labels_path)
    print(f"[B3] Loaded pose classes: {labels}")

    print("[B3] Loading MoveNet keypoint extractor...")
    movenet = load_pose_detector()

    # Open webcam
    cap = RobustVideoCapture(0)
    print("[B3] Webcam active. Press 'q' to stop.")

    # Video writer for output/b3_pose_result.mp4
    video_out_path = os.path.join('DL_PR4B', 'output', 'b3_pose_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    rep_counter = 0
    in_squat_down = False
    prev_state = "Rest"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        if video_writer is None:
            video_writer = cv2.VideoWriter(video_out_path, fourcc, 20.0, (w, h))

        # (d) Per webcam frame: run MoveNet to get 17 keypoints
        # MoveNet expects [1, 192, 192, 3] image tensor
        resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (192, 192))
        input_tensor = np.expand_dims(resized, axis=0)

        outputs = movenet(input_tensor)
        # Extract 17 keypoints [y, x, confidence]
        if isinstance(outputs, dict) and 'output_0' in outputs:
            raw_kps = outputs['output_0']
            if hasattr(raw_kps, 'numpy'):
                raw_kps = raw_kps.numpy()
            keypoints = raw_kps[0, 0, :, :]
        else:
            keypoints = np.zeros((17, 3), dtype=np.float32)

        # Flatten to (1, 34) (17 keypoints x 2 coordinates: x, y)
        flat_coords = keypoints[:, :2].flatten().reshape(1, 34)

        # Feed into pose model
        prediction = pose_model.predict(flat_coords, verbose=0)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])
        class_name = labels[class_idx] if class_idx < len(labels) else f"Pose {class_idx}"

        # (B3.5) Exercise Rep Counter: Squat Down -> Squat Up transition
        if confidence > 0.65:
            if "down" in class_name.lower():
                in_squat_down = True
            elif "up" in class_name.lower():
                if in_squat_down:
                    rep_counter += 1
                    in_squat_down = False
            prev_state = class_name

        # (e) Draw skeleton and keypoints on frame
        pixel_points = {}
        for idx in range(17):
            ky, kx, conf = keypoints[idx]
            px, py = int(kx * w), int(ky * h)
            pixel_points[idx] = (px, py, conf)
            if conf > 0.25:
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
                cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)

        for p1, p2 in SKELETON_PAIRS:
            if p1 in pixel_points and p2 in pixel_points:
                x1, y1, c1 = pixel_points[p1]
                x2, y2, c2 = pixel_points[p2]
                if c1 > 0.25 and c2 > 0.25:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # (f) Overlay UI panels: Pose class, Confidence, and Reps
        cv2.rectangle(frame, (10, 10), (360, 90), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (360, 90), (255, 180, 50), 2)

        cv2.putText(frame, f"Pose: {class_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

        # Rep Counter Badge
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 85), (0, 100, 0), -1)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 85), (255, 255, 255), 2)
        cv2.putText(frame, "SQUAT REPS", (w - 185, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"{rep_counter}", (w - 125, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Write frame to video
        video_writer.write(frame)

        cv2.imshow("Pose Classifier & Rep Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
        print(f"[B3] Output video recorded and saved to {video_out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
