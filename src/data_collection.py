"""
Data Collection Script for Indian Sign Language (ISL) Detection.
Captures hand landmarks from webcam using MediaPipe Tasks API
and saves them as numpy arrays for model training.

Supports:
  - 6 specific Word classes: WORD_HELLO, WORD_NAMASTE, WORD_I, WORD_BYE, WORD_SORRY, WORD_THANK_YOU
  - 26 Alphabet classes: LETTER_A through LETTER_Z
  - Exact total: 32 static classes
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import tkinter as tk
from PIL import Image, ImageTk

# Import configuration
try:
    from src.config import (
        STATIC_CLASSES, WORD_CLASSES, ALPHABET_CLASSES, DYNAMIC_CLASSES,
        DISPLAY_NAME_MAP, SEQUENCE_LENGTH, SAMPLES_PER_CLASS,
        ALPHABET_SAMPLES_PER_CLASS, SEQUENCES_PER_CLASS, RAW_DATA_PATH,
        MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MODELS_PATH
    )
    NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
    NUM_SEQUENCES = SEQUENCES_PER_CLASS
except ImportError:
    from config import (
        STATIC_CLASSES, WORD_CLASSES, ALPHABET_CLASSES, DYNAMIC_CLASSES,
        DISPLAY_NAME_MAP, SEQUENCE_LENGTH, SAMPLES_PER_CLASS,
        ALPHABET_SAMPLES_PER_CLASS, SEQUENCES_PER_CLASS, RAW_DATA_PATH,
        MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MODELS_PATH
    )
    NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
    NUM_SEQUENCES = SEQUENCES_PER_CLASS

# Resolve hand_landmarker.task model path
_MODEL_PATH = os.path.join(MODELS_PATH, 'hand_landmarker.task')
if not os.path.exists(_MODEL_PATH):
    _alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'hand_landmarker.task')
    if os.path.exists(_alt):
        _MODEL_PATH = _alt
    else:
        print(f"ERROR: hand_landmarker.task not found at {_MODEL_PATH}")
        exit(1)

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

def draw_hand_landmarks(image, hand_landmarks, image_width, image_height,
                        landmark_color=(0, 255, 0), connection_color=(255, 255, 255)):
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append((x, y))
        cv2.circle(image, (x, y), 4, landmark_color, -1)
        cv2.circle(image, (x, y), 6, landmark_color, 1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], connection_color, 2)

def extract_hand_landmarks(detection_result):
    if not detection_result.hand_landmarks:
        return None

    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[hand_idx][0].category_name
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

        if handedness == 'Left':
            left_hand = landmarks
        elif handedness == 'Right':
            right_hand = landmarks

    return np.concatenate([left_hand, right_hand])

def draw_visual_feedback(image, internal_class, current_count, total_count, fps,
                         recording=False, is_dynamic=False, sequence=None,
                         total_sequences=None, hand_detected=None):
    h, w, _ = image.shape
    border_color = (0, 255, 0) if (recording and hand_detected is not False) else (0, 0, 255)
    cv2.rectangle(image, (0, 0), (w, h), border_color, 4)

    # Top info background
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)

    cv2.putText(image, f"FPS: {fps}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    disp_name = DISPLAY_NAME_MAP.get(internal_class, internal_class)
    cv2.putText(image, f"Class: {disp_name} ({internal_class})", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    if recording and hand_detected is not None:
        status_text = "Hand Detected" if hand_detected else "NO HAND DETECTED"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(image, status_text, (w // 2 - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if recording:
        cv2.circle(image, (w - 30, h - 30), 10, (0, 0, 255), -1)
        cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)

        if is_dynamic:
            info_text = f"Sequence: {sequence}/{total_sequences} | Frames: {current_count}/{total_count}"
        else:
            info_text = f"Collected: {current_count}/{total_count}"

        cv2.putText(image, info_text, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(image, "Press 'S' to Start Recording | 'Q' to Quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image

def show_countdown(cap, root=None, tk_label=None):
    for i in range(3, 0, -1):
        start_t = time.time()
        while time.time() - start_t < 1.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            cv2.putText(frame, str(i), (w // 2 - 20, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 4)
            if root is not None and tk_label is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                tk_label.configure(image=img_tk)
                tk_label.image = img_tk
                root.update()
            else:
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)
            time.sleep(0.03)

def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    return vision.HandLandmarker.create_from_options(options)

def collect_static_data(internal_class, num_samples=NUM_SAMPLES_STATIC):
    """
    Collects static hand sign data for an internal class (e.g., WORD_HELLO or LETTER_A).
    Saves strictly to data/raw/{internal_class}/static_data.npy.
    """
    save_dir = os.path.join(RAW_DATA_PATH, internal_class)
    os.makedirs(save_dir, exist_ok=True)
    existing_path = os.path.join(save_dir, "static_data.npy")
    existing_count = 0
    if os.path.exists(existing_path):
        try:
            existing_count = np.load(existing_path).shape[0]
        except Exception:
            pass

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    disp_name = DISPLAY_NAME_MAP.get(internal_class, internal_class)
    print(f"\n--- Collecting static data for '{disp_name}' [Folder: {internal_class}] ---")
    print(f"Target: {num_samples} NEW samples | Already saved: {existing_count}")
    print("Position hand gesture. Press 'S' to start recording, 'Q' to cancel.")

    data = []

    # Initialize Tkinter preview window
    root = tk.Tk()
    root.title(f"Data Collection - {disp_name}")
    root.geometry("800x620")
    root.attributes("-topmost", True)

    img_label = tk.Label(root, bg="black")
    img_label.pack(fill="both", expand=True)

    data = []
    state = {
        "recording": False,
        "quit": False,
        "p_time": 0,
        "frame_count": 0
    }

    def on_key(event):
        char = event.char.lower()
        if char == 'q':
            state["quit"] = True
        elif char == 's' and not state["recording"]:
            show_countdown(cap, root=root, tk_label=img_label)
            state["recording"] = True
            state["p_time"] = time.time()

    def on_close():
        state["quit"] = True

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.focus_force()

    landmarker = create_landmarker()

    while cap.isOpened() and not state["quit"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        state["frame_count"] += 1
        timestamp_ms = int(state["frame_count"] * 1000 / 30)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, w, h)

        c_time = time.time()
        fps = int(1 / (c_time - state["p_time"])) if (c_time - state["p_time"]) > 0 else 0
        state["p_time"] = c_time

        if state["recording"]:
            landmarks = extract_hand_landmarks(detection_result)
            if landmarks is not None:
                data.append(landmarks)

            if len(data) >= num_samples:
                print(f"Successfully collected {num_samples} samples for {internal_class}.")
                break

        hand_detected = extract_hand_landmarks(detection_result) is not None
        frame = draw_visual_feedback(frame, internal_class, len(data), num_samples,
                                     fps, state["recording"], hand_detected=hand_detected)

        # Render frame to Tkinter preview using PIL
        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_disp)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        img_label.configure(image=tk_img)
        img_label.image = tk_img

        try:
            root.update()
        except tk.TclError:
            break

    landmarker.close()
    cap.release()
    try:
        root.destroy()
    except Exception:
        pass

    if len(data) > 0:
        np_new = np.array(data)
        if os.path.exists(existing_path) and existing_count > 0:
            existing_data = np.load(existing_path)
            combined = np.concatenate([existing_data, np_new], axis=0)
            np.save(existing_path, combined)
            print(f"Appended {len(np_new)} new samples => Total {combined.shape[0]} samples saved to {existing_path}")
        else:
            np.save(existing_path, np_new)
            print(f"Data saved to {existing_path} with shape {np_new.shape}")

def collect_dynamic_data(class_name, num_sequences=NUM_SEQUENCES, sequence_length=SEQUENCE_LENGTH):
    """Dynamic gesture capture preserved untouched."""
    save_dir = os.path.join(RAW_DATA_PATH, class_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    data = []
    landmarker = create_landmarker()
    print(f"\n--- Collecting dynamic data for '{class_name}' ---")
    print("Press 'S' to start recording sequences, 'Q' to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Dynamic: {class_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'S' to Start, 'Q' to Quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            landmarker.close()
            cap.release()
            cv2.destroyAllWindows()
            return

    p_time = 0
    for sequence in range(1, num_sequences + 1):
        sequence_data = []
        for i in range(2, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sequence {sequence} in {i}...",
                        (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        frame_num = 0
        while frame_num < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            frame_count += 1
            timestamp_ms = int(frame_count * 1000 / 30)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks, w, h)

            c_time = time.time()
            fps = int(1 / (c_time - p_time)) if (c_time - p_time) > 0 else 0
            p_time = c_time

            landmarks = extract_hand_landmarks(detection_result)
            hand_present = landmarks is not None
            if hand_present:
                sequence_data.append(landmarks)
                frame_num += 1

            frame = draw_visual_feedback(
                frame, class_name, frame_num, sequence_length, fps,
                recording=True, is_dynamic=True, sequence=sequence,
                total_sequences=num_sequences, hand_detected=hand_present
            )
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                landmarker.close()
                cap.release()
                cv2.destroyAllWindows()
                return

        data.append(sequence_data)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    if len(data) == 0:
        return

    np_new = np.array(data)
    save_path = os.path.join(save_dir, "dynamic_data.npy")
    if os.path.exists(save_path):
        existing = np.load(save_path)
        combined = np.concatenate([existing, np_new], axis=0)
        np.save(save_path, combined)
    else:
        np.save(save_path, np_new)

def collect_alphabet_session():
    print("\n" + "=" * 50)
    print("  ISL ALPHABET DATA COLLECTION (LETTER_A to LETTER_Z)")
    print("=" * 50)
    print("Options:")
    print("  - Type any letter (e.g. A, B, C, ..., Z)")
    print("  - Type 'ALL' to collect all 26 letters sequentially")
    print("  - Type 'BACK' to return")

    while True:
        cmd = input("\nEnter letter (A-Z) or command: ").strip().upper()
        if cmd == 'BACK':
            break
        elif cmd == 'ALL':
            for letter in [chr(c) for c in range(ord('A'), ord('Z') + 1)]:
                internal_cls = f"LETTER_{letter}"
                print(f"\n>>> Preparing: {internal_cls} ({letter}) <<<")
                input("Press ENTER when ready, then press S in the camera window...")
                collect_static_data(internal_cls, num_samples=ALPHABET_SAMPLES_PER_CLASS)
        elif len(cmd) == 1 and 'A' <= cmd <= 'Z':
            internal_cls = f"LETTER_{cmd}"
            collect_static_data(internal_cls, num_samples=ALPHABET_SAMPLES_PER_CLASS)
        else:
            print("Invalid input. Please enter A-Z, ALL, or BACK.")

def show_summary():
    print("\n" + "=" * 55)
    print("       32-CLASS DATASET STATUS SUMMARY")
    print("=" * 55)
    print("1. WORD CLASSES (6 target classes):")
    for wc in WORD_CLASSES:
        p = os.path.join(RAW_DATA_PATH, wc, "static_data.npy")
        cnt = np.load(p).shape[0] if os.path.exists(p) else 0
        disp = DISPLAY_NAME_MAP[wc]
        mark = "✓" if cnt >= 200 else " "
        print(f"  [{mark}] {wc:18s} ({disp:10s}): {cnt:4d} samples")

    print("\n2. ALPHABET CLASSES (26 target classes):")
    total_letters = 0
    for ac in ALPHABET_CLASSES:
        p = os.path.join(RAW_DATA_PATH, ac, "static_data.npy")
        cnt = np.load(p).shape[0] if os.path.exists(p) else 0
        total_letters += cnt
        disp = DISPLAY_NAME_MAP[ac]
        mark = "✓" if cnt >= 200 else " "
        print(f"  [{mark}] {ac:18s} ({disp}): {cnt:4d} samples")

    print(f"\nTotal Alphabet Samples Collected: {total_letters}")
    print("=" * 55 + "\n")

def main():
    while True:
        print("\n--- ISL 32-Class Data Collection Menu ---")
        print("1. Collect Static Data for a specific WORD (from the 6 required)")
        print("2. Collect Static Data for ALL 6 Words sequentially")
        print("3. Collect Static Data for ALPHABETS (LETTER_A to LETTER_Z)")
        print("4. View 32-Class Dataset Collection Summary")
        print("5. Collect Dynamic Data (preserved for dynamic signs)")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()
        if choice == '1':
            print("\nAvailable Words:")
            for i, wc in enumerate(WORD_CLASSES, 1):
                print(f"  {i}. {wc} ({DISPLAY_NAME_MAP[wc]})")
            w_choice = input("Enter word name or number: ").strip()
            if w_choice.isdigit() and 1 <= int(w_choice) <= len(WORD_CLASSES):
                selected = WORD_CLASSES[int(w_choice) - 1]
                collect_static_data(selected, num_samples=SAMPLES_PER_CLASS)
            elif w_choice.upper() in WORD_CLASSES:
                collect_static_data(w_choice.upper(), num_samples=SAMPLES_PER_CLASS)
            else:
                matched = [wc for wc in WORD_CLASSES if DISPLAY_NAME_MAP[wc].lower() == w_choice.lower()]
                if matched:
                    collect_static_data(matched[0], num_samples=SAMPLES_PER_CLASS)
                else:
                    print("Class not found.")
        elif choice == '2':
            for wc in WORD_CLASSES:
                print(f"\n>>> Preparing: {wc} ({DISPLAY_NAME_MAP[wc]}) <<<")
                input("Press ENTER when ready, then press S in the camera window...")
                collect_static_data(wc, num_samples=SAMPLES_PER_CLASS)
        elif choice == '3':
            collect_alphabet_session()
        elif choice == '4':
            show_summary()
        elif choice == '5':
            print("\nAvailable Dynamic Classes:", DYNAMIC_CLASSES)
            cls = input("Enter dynamic class name: ").strip()
            if cls in DYNAMIC_CLASSES:
                collect_dynamic_data(cls)
            else:
                print("Invalid dynamic class.")
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
