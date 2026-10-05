"""
Real-time Indian Sign Language recognition and translation.
Captures webcam feed, extracts hand landmarks using MediaPipe Tasks API,
classifies gestures, displays predictions, and speaks detected signs using TTS.
"""

import os
import sys
from unittest.mock import MagicMock

# Mock matplotlib to prevent DLL load failure under Windows Application Control policy
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.ft2font'] = MagicMock()

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from collections import deque

try:
    from src import config
    from src.tts import TTSEngine
    from src import llm_translator
except ImportError:
    try:
        import config
        from tts import TTSEngine
        import llm_translator
    except ImportError:
        print("Error: Could not import config, tts, or llm_translator modules.")
        print("Please run from the project root or src directory.")
        exit(1)

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_landmarks(image, hand_landmarks, image_width, image_height,
                        landmark_color=(0, 255, 0), connection_color=(255, 255, 255)):
    """Draw hand landmarks and connections on an image."""
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


class ISLRecognizer:
    """
    Real-time Indian Sign Language recognizer.
    Supports static (single-frame) and dynamic (sequence-based) modes.
    """

    def __init__(self, mode='static'):
        self.mode = mode
        self.sequence_length = getattr(config, 'SEQUENCE_LENGTH', 30)

        # Resolve model path for MediaPipe
        models_path = getattr(config, 'MODELS_PATH',
                              str(getattr(config, 'MODELS_DIR', 'models')))
        self.mp_model_path = os.path.join(models_path, 'hand_landmarker.task')
        if not os.path.exists(self.mp_model_path):
            print(f"ERROR: hand_landmarker.task not found at {self.mp_model_path}")
            print("Download from: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            exit(1)

        # Initialize MediaPipe HandLandmarker (VIDEO mode)
        base_options = python.BaseOptions(model_asset_path=self.mp_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.7),
            min_hand_presence_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.7),
            min_tracking_confidence=getattr(config, 'MIN_TRACKING_CONFIDENCE', 0.5)
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Initialize TTS
        self.tts = TTSEngine()

        # Load ML Models and Encoders
        self.static_model = None
        self.dynamic_model = None
        self.static_classes = None
        self.dynamic_classes = None
        self.load_models()

        # Prediction stabilization
        stability_frames = getattr(config, 'STABILITY_FRAMES', 10)
        self.prediction_buffer = deque(maxlen=stability_frames)
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        # Sentence builder state
        self.current_sentence = []
        self.last_spoken_time = 0
        self.cooldown_period = 1.0
        self.last_word = ""

        # Phase 1: Gesture lifecycle & debounce state
        # Tracks gesture strokes: prevents duplicate additions while holding,
        # and resets state only when hands rest or gesture concludes.
        self.active_gesture = None          # Current active sign class name (e.g. 'Hello')
        self.gesture_committed = False      # True once current stroke has been added to current_sentence
        self.neutral_frame_count = 0        # Counter for consecutive neutral/unconfirmed frames
        self.reset_frames = getattr(config, 'GESTURE_RESET_FRAMES', 8)
        self.rest_y_threshold = getattr(config, 'REST_POSITION_Y_THRESHOLD', 0.85)

        # Frame counter for timestamps
        self.frame_count = 0

    def load_models(self):
        """Load trained ML models and label encoders based on mode."""
        proc_path = getattr(config, 'PROCESSED_DATA_PATH',
                            str(getattr(config, 'DATA_PROCESSED_DIR', 'data/processed')))
        models_path = getattr(config, 'MODELS_PATH',
                              str(getattr(config, 'MODELS_DIR', 'models')))

        try:
            if self.mode in ['static', 'both']:
                static_model_path = os.path.join(models_path, 'static_model_6class.h5')
                static_le_path = os.path.join(proc_path, 'label_encoder_6class.npy')

                if not os.path.exists(static_model_path):
                    raise FileNotFoundError(
                        f"Static model not found at {static_model_path}. "
                        "Please train the model first: python src/train_static.py"
                    )
                if not os.path.exists(static_le_path):
                    raise FileNotFoundError(
                        f"Label encoder not found at {static_le_path}. "
                        "Please run preprocessing first: python src/preprocess.py"
                    )

                self.static_model = tf.keras.models.load_model(static_model_path)
                self.static_classes = np.load(static_le_path, allow_pickle=True)
                print(f"  Static model loaded ({len(self.static_classes)} classes)")

            if self.mode in ['dynamic', 'both']:
                dynamic_model_path = os.path.join(models_path, 'dynamic_model.h5')
                dynamic_le_path = os.path.join(proc_path, 'label_encoder_dynamic.npy')

                if not os.path.exists(dynamic_model_path):
                    raise FileNotFoundError(
                        f"Dynamic model not found at {dynamic_model_path}. "
                        "Please train the model first: python src/train_dynamic.py"
                    )
                if not os.path.exists(dynamic_le_path):
                    raise FileNotFoundError(
                        f"Dynamic label encoder not found at {dynamic_le_path}."
                    )

                self.dynamic_model = tf.keras.models.load_model(dynamic_model_path)
                self.dynamic_classes = np.load(dynamic_le_path, allow_pickle=True)
                print(f"  Dynamic model loaded ({len(self.dynamic_classes)} classes)")

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nPlease follow these steps:")
            print("  1. Collect data:    python src/data_collection.py")
            print("  2. Preprocess:      python src/preprocess.py")
            print("  3. Train model:     python src/train_static.py")
            print("  4. Run inference:   python src/inference.py")
            exit(1)
        except Exception as e:
            print(f"\nUnexpected error loading models: {e}")
            exit(1)

    def extract_landmarks(self, frame):
        """
        Process a frame with MediaPipe Tasks API and extract hand landmarks.

        Returns:
            landmarks: numpy array of shape (126,)
            frame: annotated frame with hand skeleton drawn
            hands_detected: boolean
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self.frame_count += 1
        timestamp_ms = int(self.frame_count * 1000 / 30)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = np.zeros(126)
        hands_detected = bool(detection_result.hand_landmarks)

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Draw on frame
                draw_hand_landmarks(frame, hand_landmarks, w, h)

                # Determine hand offset
                handedness = detection_result.handedness[idx][0].category_name
                offset = 63 if handedness == 'Right' else 0

                # Extract landmark coordinates
                for i, lm in enumerate(hand_landmarks):
                    landmarks[offset + i * 3] = lm.x
                    landmarks[offset + i * 3 + 1] = lm.y
                    landmarks[offset + i * 3 + 2] = lm.z

        return landmarks, frame, hands_detected

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position for each hand."""
        normalized = np.zeros_like(landmarks)
        for i in range(0, 126, 63):
            hand = landmarks[i:i + 63]
            if np.all(hand == 0):
                continue
            wrist = hand[0:3]
            shifted = hand.copy()
            for j in range(0, 63, 3):
                shifted[j] -= wrist[0]
                shifted[j + 1] -= wrist[1]
                shifted[j + 2] -= wrist[2]
            distances = [np.sqrt(shifted[j]**2 + shifted[j+1]**2 + shifted[j+2]**2)
                         for j in range(0, 63, 3)]
            max_dist = max(distances) if distances else 0
            if max_dist > 0:
                shifted = shifted / max_dist
            normalized[i:i + 63] = shifted
        return normalized

    def predict_static(self, landmarks):
        """Predict sign from single-frame landmarks."""
        normalized = self.normalize_landmarks(landmarks)
        pred = self.static_model.predict(np.array([normalized]), verbose=0)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.static_classes[class_idx]
        return class_name, confidence

    def predict_dynamic(self, sequence):
        """Predict sign from a sequence of landmark frames."""
        seq_array = np.array(sequence)
        # Normalize each frame in the sequence
        normalized_seq = np.array([self.normalize_landmarks(frame) for frame in seq_array])
        pred = self.dynamic_model.predict(np.array([normalized_seq]), verbose=0)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.dynamic_classes[class_idx]
        return class_name, confidence

    def stabilize_prediction(self, prediction, confidence):
        """Only confirm if same prediction appears consistently with high confidence."""
        threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.7)
        if confidence > threshold:
            self.prediction_buffer.append(prediction)
        else:
            self.prediction_buffer.append(None)

        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            if (len(set(self.prediction_buffer)) == 1
                    and self.prediction_buffer[0] is not None):
                return self.prediction_buffer[0]
        return None

    def is_hand_at_rest(self, landmarks):
        """
        Check if all detected hands are resting near the bottom of the camera frame.
        Uses normalized MediaPipe landmark coordinates where y=0 is top and y=1 is bottom.
        """
        wrist_ys = []
        # Left hand wrist (offset 0, landmark index 0 -> y coordinate at index 1)
        if np.any(landmarks[0:63] != 0):
            wrist_ys.append(landmarks[1])
        # Right hand wrist (offset 63, landmark index 0 -> y coordinate at index 64)
        if np.any(landmarks[63:126] != 0):
            wrist_ys.append(landmarks[64])

        if not wrist_ys:
            return True
        return all(y > self.rest_y_threshold for y in wrist_ys)

    def _handle_gesture_commit(self, stable_pred, current_conf=0.0):
        """
        Commit a stable prediction as a single gesture stroke.
        Prevents duplicate additions while holding the same sign.
        Allows immediate transition if the sign changes.
        """
        if stable_pred is not None:
            self.neutral_frame_count = 0
            # Commit if no gesture is currently active, or if sign transitioned to a new class
            if not self.gesture_committed or stable_pred != self.active_gesture:
                self.tts.speak(stable_pred)
                self.current_sentence.append(stable_pred)
                self.last_word = stable_pred
                self.active_gesture = stable_pred
                self.gesture_committed = True
                self.last_spoken_time = time.time()
                return True
        return False

    def _handle_gesture_reset(self):
        """
        Increment neutral frame counter and reset the gesture stroke state
        once hands rest, leave the frame, or confidence drops for consecutive frames.
        """
        self.neutral_frame_count += 1
        if self.neutral_frame_count >= self.reset_frames:
            self.active_gesture = None
            self.gesture_committed = False

    def clear_sentence(self):
        """Clear accumulated sentence and reset gesture lifecycle state."""
        self.current_sentence = []
        self.last_word = ""
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0

    def process_frame(self, frame):
        """
        Process a single frame for GUI integration.
        Returns (annotated_frame, prediction, confidence, hands_detected).
        The old run() method is preserved for CLI usage.
        """
        frame = cv2.flip(frame, 1)
        landmarks, annotated_frame, hands_detected = self.extract_landmarks(frame)

        current_pred = None
        current_conf = 0.0
        stable_pred = None

        is_rest = not hands_detected or self.is_hand_at_rest(landmarks)

        if hands_detected and not is_rest:
            if self.mode == 'static' and self.static_model is not None:
                raw_pred, conf = self.predict_static(landmarks)
                current_pred = raw_pred
                current_conf = conf
            elif self.mode == 'dynamic' and self.dynamic_model is not None:
                self.sequence_buffer.append(landmarks)
                if len(self.sequence_buffer) == self.sequence_length:
                    raw_pred, conf = self.predict_dynamic(list(self.sequence_buffer))
                    current_pred = raw_pred
                    current_conf = conf

            stable_pred = self.stabilize_prediction(current_pred, current_conf)

            if stable_pred:
                self._handle_gesture_commit(stable_pred, current_conf)
            else:
                self._handle_gesture_reset()
        else:
            if self.mode == 'dynamic' and len(self.sequence_buffer) > 0:
                self.sequence_buffer.popleft()
            self.prediction_buffer.append(None)
            self._handle_gesture_reset()

        return annotated_frame, current_pred, current_conf, hands_detected

    def swap_static_model(self, model_path: str, le_path: str) -> tuple[bool, str]:
        """
        Hot-swap the static model and its label encoder at runtime.
        MediaPipe, TTS, and dynamic model are completely untouched.

        Returns:
            (success, message)
        """
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        if not os.path.exists(le_path):
            return False, f"Label encoder not found: {le_path}"
        try:
            new_model = tf.keras.models.load_model(model_path)
            new_classes = np.load(le_path, allow_pickle=True)
            self.static_model = new_model
            self.static_classes = new_classes
            # Clear prediction state so stale buffers don't bleed across models
            self.prediction_buffer.clear()
            self.active_gesture = None
            self.gesture_committed = False
            self.neutral_frame_count = 0
            print(f"  Static model swapped: {os.path.basename(model_path)} "
                  f"({len(new_classes)} classes)")
            return True, f"Loaded {os.path.basename(model_path)} ({len(new_classes)} classes)"
        except Exception as e:
            return False, f"Failed to load model: {e}"

    def set_mode(self, new_mode):
        """Switch recognition mode at runtime. Used by the GUI app."""
        if new_mode == self.mode:
            return
        self.mode = new_mode
        self.prediction_buffer.clear()
        self.sequence_buffer.clear()
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0

    def draw_ui(self, frame, prediction, confidence, fps):
        """Draw polished UI overlay on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent panels
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Prediction
        pred_text = prediction if prediction else "--"
        cv2.putText(frame, f"Sign: {pred_text}", (20, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if confidence and confidence > 0:
            bar_x = 350
            bar_w = int(200 * confidence)
            bar_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 50), bar_color, -1)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + 200, 50), (255, 255, 255), 1)
            cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + 210, 48),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mode and FPS
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (w - 220, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (w - 220, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Sentence and controls
        sentence_str = " ".join(self.current_sentence) if self.current_sentence else "(empty)"
        cv2.putText(frame, f"Sentence: {sentence_str}", (20, h - 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        controls = "C: Clear | Space: Speak | M: Mode | Q: Quit"
        cv2.putText(frame, controls, (20, h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame

    def run(self):
        """Main inference loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        pTime = 0

        print(f"\n{'='*50}")
        print(f"  ISL Recognition System - {self.mode.upper()} mode")
        print(f"{'='*50}")
        print("Controls: Q=Quit  C=Clear  Space=Speak  M=Mode")
        print(f"{'='*50}\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame, hands_detected = self.extract_landmarks(frame)

            current_pred = None
            current_conf = 0.0

            is_rest = not hands_detected or self.is_hand_at_rest(landmarks)

            if hands_detected and not is_rest:
                if self.mode == 'static' and self.static_model is not None:
                    raw_pred, conf = self.predict_static(landmarks)
                    current_pred = raw_pred
                    current_conf = conf
                elif self.mode == 'dynamic' and self.dynamic_model is not None:
                    self.sequence_buffer.append(landmarks)
                    if len(self.sequence_buffer) == self.sequence_length:
                        raw_pred, conf = self.predict_dynamic(list(self.sequence_buffer))
                        current_pred = raw_pred
                        current_conf = conf

                stable_pred = self.stabilize_prediction(current_pred, current_conf)

                if stable_pred:
                    if self._handle_gesture_commit(stable_pred, current_conf):
                        print(f"  Detected: {stable_pred} ({current_conf*100:.0f}%)")
                else:
                    self._handle_gesture_reset()
            else:
                if self.mode == 'dynamic' and len(self.sequence_buffer) > 0:
                    self.sequence_buffer.popleft()
                self.prediction_buffer.append(None)
                self._handle_gesture_reset()

            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if pTime > 0 else 0
            pTime = cTime

            output_frame = self.draw_ui(annotated_frame, current_pred, current_conf, fps)
            cv2.imshow('ISL Recognition System', output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_sentence()
                print("  Sentence cleared.")
            elif key == ord(' '):
                if self.current_sentence:
                    sentence = llm_translator.correct_grammar(self.current_sentence)
                    print(f"  Speaking: \"{sentence}\"")
                    self.tts.speak(sentence)
            elif key == ord('m'):
                if self.static_model and self.dynamic_model:
                    new_mode = 'dynamic' if self.mode == 'static' else 'static'
                    self.set_mode(new_mode)
                    print(f"  Switched to {self.mode.upper()} mode")
                else:
                    print("  Cannot toggle: both models must be loaded (use --mode both)")

        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nISL Recognition System closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ISL Recognition & Translation")
    parser.add_argument("--mode", type=str, default="static",
                        choices=['static', 'dynamic', 'both'],
                        help="Recognition mode: static, dynamic, or both")
    args = parser.parse_args()

    recognizer = ISLRecognizer(mode=args.mode)
    recognizer.run()
