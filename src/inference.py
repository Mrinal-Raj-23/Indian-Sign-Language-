"""
Real-time Indian Sign Language recognition and translation.
Captures webcam feed, extracts hand landmarks using MediaPipe Tasks API,
classifies gestures across 32 static classes (6 words + 26 alphabets),
buffers letters into words, applies pause detection and pyspellchecker correction,
and speaks detected sentences via TTS.
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
    import config
    from tts import TTSEngine
    import llm_translator

# Optional: pyspellchecker
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    SPELLCHECK_AVAILABLE = True
except ImportError:
    _spell = None
    SPELLCHECK_AVAILABLE = False

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

def spell_check_word(candidate: str) -> str:
    if not SPELLCHECK_AVAILABLE or not candidate or len(candidate) <= 1:
        return candidate
    lower = candidate.lower()
    if lower in _spell:
        return candidate
    correction = _spell.correction(lower)
    if correction and correction != lower:
        print(f"  [SpellCheck] '{candidate}' -> '{correction}'")
        return correction.upper() if candidate.isupper() else correction.capitalize()
    return candidate

class ISLRecognizer:
    def __init__(self, mode='static'):
        self.mode = mode
        self.sequence_length = getattr(config, 'SEQUENCE_LENGTH', 30)

        models_path = getattr(config, 'MODELS_PATH', str(getattr(config, 'MODELS_DIR', 'models')))
        self.mp_model_path = os.path.join(models_path, 'hand_landmarker.task')
        if not os.path.exists(self.mp_model_path):
            print(f"ERROR: hand_landmarker.task not found at {self.mp_model_path}")
            exit(1)

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
        self.tts = TTSEngine()

        self.static_model = None
        self.dynamic_model = None
        self.static_classes = None
        self.dynamic_classes = None
        self._last_static_probabilities = None
        self.load_models()

        self.display_map = getattr(config, 'DISPLAY_NAME_MAP', {})
        self.alphabet_classes = getattr(config, 'ALPHABET_CLASS_SET', set())

        # Prediction stabilization
        stability_frames = getattr(config, 'STABILITY_FRAMES', 5)
        self.prediction_buffer = deque(maxlen=stability_frames)
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        # Sentence builder state
        self.current_sentence = []
        self.last_spoken_time = 0
        self.last_word = ""

        # Gesture debouncing
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0
        self.reset_frames = getattr(config, 'GESTURE_RESET_FRAMES', 8)
        self.rest_y_threshold = getattr(config, 'REST_POSITION_Y_THRESHOLD', 0.85)

        # Fingerspelling buffering
        self._letter_stability_frames = getattr(config, 'LETTER_STABILITY_FRAMES', 10)
        self._inter_letter_gap = getattr(config, 'INTER_LETTER_GAP_SECONDS', 0.8)
        self._word_end_pause   = max(getattr(config, 'WORD_END_PAUSE_SECONDS', 2.0), 3.5)
        self._letter_buffer = []
        self._last_committed_letter = ""
        self._current_normalized_pose = None
        self._last_committed_pose = None
        self._letter_stable_count = 0
        self._candidate_letter = ""
        self._pending_letter = ""
        self._pending_letter_since = None
        self._letter_confirmation_seconds = 0.2
        self._static_hand_active = False
        self._static_hand_settle_until = 0.0
        self._static_hand_settle_seconds = 0.4
        self._last_gesture_time = time.time()
        self._word_finalised_for_this_pause = False

        self.frame_count = 0

    def load_models(self):
        proc_path = getattr(config, 'PROCESSED_DATA_PATH', str(getattr(config, 'DATA_PROCESSED_DIR', 'data/processed')))
        models_path = getattr(config, 'MODELS_PATH', str(getattr(config, 'MODELS_DIR', 'models')))

        try:
            if self.mode in ['static', 'both']:
                model_32 = os.path.join(models_path, 'static_model_32class.h5')
                le_32    = os.path.join(proc_path,   'label_encoder_32class.npy')
                model_6  = os.path.join(models_path, 'static_model_6class.h5')
                le_6     = os.path.join(proc_path,   'label_encoder_6class.npy')

                if os.path.exists(model_32) and os.path.exists(le_32):
                    self.static_model = tf.keras.models.load_model(model_32)
                    self.static_classes = np.load(le_32, allow_pickle=True)
                    print(f"Loaded 32-Class Static Model: {len(self.static_classes)} classes.")
                elif os.path.exists(model_6) and os.path.exists(le_6):
                    self.static_model = tf.keras.models.load_model(model_6)
                    self.static_classes = np.load(le_6, allow_pickle=True)
                    print(f"Loaded Legacy Static Model ({len(self.static_classes)} classes).")
                else:
                    raise FileNotFoundError("Static model not found. Please train static model first.")

            if self.mode in ['dynamic', 'both']:
                dyn_model_path = os.path.join(models_path, 'dynamic_model.h5')
                dyn_le_path    = os.path.join(proc_path,   'label_encoder_dynamic.npy')
                if os.path.exists(dyn_model_path) and os.path.exists(dyn_le_path):
                    self.dynamic_model = tf.keras.models.load_model(dyn_model_path)
                    self.dynamic_classes = np.load(dyn_le_path, allow_pickle=True)
                    print(f"Loaded Dynamic Model ({len(self.dynamic_classes)} classes).")
                else:
                    print("Dynamic model files not found; dynamic mode unavailable.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def extract_landmarks(self, frame):
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
                draw_hand_landmarks(frame, hand_landmarks, w, h)
                handedness = detection_result.handedness[idx][0].category_name
                offset = 63 if handedness == 'Right' else 0
                for i, lm in enumerate(hand_landmarks):
                    landmarks[offset + i * 3]     = lm.x
                    landmarks[offset + i * 3 + 1] = lm.y
                    landmarks[offset + i * 3 + 2] = lm.z

        return landmarks, frame, hands_detected

    def normalize_landmarks(self, landmarks):
        normalized = np.zeros_like(landmarks)
        for i in range(0, 126, 63):
            hand = landmarks[i:i + 63]
            if np.all(hand == 0):
                continue
            wrist = hand[0:3]
            shifted = hand.copy()
            for j in range(0, 63, 3):
                shifted[j]     -= wrist[0]
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
        normalized = self.normalize_landmarks(landmarks)
        self._current_normalized_pose = normalized
        pred = self.static_model.predict_on_batch(np.array([normalized]))[0]
        self._last_static_probabilities = pred
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.static_classes[class_idx]
        return class_name, confidence

    def predict_dynamic(self, sequence):
        seq_array = np.array(sequence)
        normalized_seq = np.array([self.normalize_landmarks(f) for f in seq_array])
        pred = self.dynamic_model.predict(np.array([normalized_seq]), verbose=0)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        class_name = self.dynamic_classes[class_idx]
        return class_name, confidence

    def stabilize_prediction(self, prediction, confidence):
        threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.6)
        previous_prediction = self.prediction_buffer[-1] if self.prediction_buffer else None
        is_eligible_letter = confidence > threshold and (
            prediction.startswith('LETTER_') or prediction in self.alphabet_classes
        )
        if self._pending_letter and (
            not is_eligible_letter
            or prediction != self._pending_letter
            or previous_prediction is None
        ):
            self._pending_letter = ""
            self._pending_letter_since = None

        if confidence > threshold:
            self.prediction_buffer.append(prediction)
            if is_eligible_letter:
                if time.monotonic() < self._static_hand_settle_until:
                    self._candidate_letter = ""
                    self._letter_stable_count = 0
                    self._pending_letter = ""
                    self._pending_letter_since = None
                    return None
                if previous_prediction is None or self._candidate_letter != prediction:
                    self._candidate_letter = prediction
                    self._letter_stable_count = 1
                else:
                    self._letter_stable_count += 1
            else:
                self._candidate_letter = ""
                self._letter_stable_count = 0
        else:
            self.prediction_buffer.append(None)
            self._candidate_letter = ""
            self._letter_stable_count = 0

        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            if len(set(self.prediction_buffer)) == 1 and self.prediction_buffer[0] is not None:
                stable_prediction = self.prediction_buffer[0]
                if stable_prediction.startswith('LETTER_') or stable_prediction in self.alphabet_classes:
                    if self._letter_stable_count < self._letter_stability_frames:
                        return None
                    if self._pending_letter != stable_prediction:
                        self._pending_letter = stable_prediction
                        self._pending_letter_since = time.monotonic()
                        return None
                    if time.monotonic() - self._pending_letter_since >= self._letter_confirmation_seconds:
                        return stable_prediction
                else:
                    return stable_prediction
        return None

    def is_hand_at_rest(self, landmarks):
        wrist_ys = []
        if np.any(landmarks[0:63] != 0):
            wrist_ys.append(landmarks[1])
        if np.any(landmarks[63:126] != 0):
            wrist_ys.append(landmarks[64])
        if not wrist_ys:
            return True
        return all(y > self.rest_y_threshold for y in wrist_ys)

    def _handle_gesture_commit(self, stable_pred, current_conf=0.0):
        if stable_pred is None:
            return False

        self.neutral_frame_count = 0
        self._last_gesture_time = time.time()
        self._word_finalised_for_this_pause = False

        # Check if Alphabet class (LETTER_*)
        if stable_pred.startswith('LETTER_') or stable_pred in self.alphabet_classes:
            char = self.display_map.get(stable_pred, stable_pred.replace('LETTER_', ''))
            return self._commit_letter(char)

        # Otherwise it's a Word class (WORD_*)
        if self._letter_buffer:
            self._finalise_letter_word()

        word_display = self.display_map.get(stable_pred, stable_pred)
        if not self.gesture_committed or stable_pred != self.active_gesture:
            self.tts.speak(word_display)
            self.current_sentence.append(word_display)
            self.last_word = word_display
            self.active_gesture = stable_pred
            self.gesture_committed = True
            return True
        return False

    def _commit_letter(self, letter: str) -> bool:
        # stabilize_prediction() already guarantees 5 consecutive matching frames;
        # no second stability gate needed here.
        if self._last_committed_letter:
            return False

        current_pose = self._current_normalized_pose
        if self.mode == 'static':
            if current_pose is None:
                return False
            if self._last_committed_letter:
                if self._last_committed_pose is None:
                    return False
                current_points = current_pose.reshape(-1, 3)
                committed_points = self._last_committed_pose.reshape(-1, 3)
                present_points = np.any(current_points != 0, axis=1) | np.any(committed_points != 0, axis=1)
                if not np.any(present_points):
                    return False
                pose_change = np.mean(
                    np.linalg.norm(current_points[present_points] - committed_points[present_points], axis=1)
                )
                threshold = getattr(config, 'LETTER_POSE_CHANGE_THRESHOLD', 0.10)
                if pose_change < threshold:
                    return False

        self._letter_buffer.append(letter)
        self._last_committed_letter = letter
        self._last_committed_pose = current_pose.copy() if current_pose is not None else None
        print(f"  [Fingerspell] Added: '{letter}' -> Current buffer: {''.join(self._letter_buffer)}")
        return True

    def _finalise_letter_word(self):
        if not self._letter_buffer:
            return
        raw_word = "".join(self._letter_buffer)
        final_word = spell_check_word(raw_word)

        print(f"  [Word Finalized] '{raw_word}' -> '{final_word}'")
        self.current_sentence.append(final_word)
        self.last_word = final_word
        self.tts.speak(final_word)

        self._letter_buffer = []
        self._last_committed_letter = ""
        self._last_committed_pose = None
        self._candidate_letter = ""
        self._letter_stable_count = 0
        self._word_finalised_for_this_pause = True

    def _check_letter_pause(self):
        if not self._letter_buffer or self._word_finalised_for_this_pause:
            return
        elapsed = time.time() - self._last_gesture_time
        # After inter-letter gap: unlock same-letter re-commit (e.g. spelling "ADD")
        
        # After word-end pause: finalize accumulated letters as one word
        if elapsed >= self._word_end_pause:
            self._finalise_letter_word()

    def _handle_gesture_reset(self):
        self.neutral_frame_count += 1
        if self.neutral_frame_count >= self.reset_frames:
            self.active_gesture = None
            self.gesture_committed = False
            # Reset _last_committed_letter so the next stable prediction of the
            # same letter after a neutral gap can commit again.
            self._last_committed_letter = ""
            self._last_committed_pose = None

    def clear_sentence(self):
        self.current_sentence = []
        self.last_word = ""
        self.active_gesture = None
        self.gesture_committed = False
        self.neutral_frame_count = 0
        self._letter_buffer = []
        self._last_committed_letter = ""
        self._last_committed_pose = None
        self._candidate_letter = ""
        self._letter_stable_count = 0
        self._word_finalised_for_this_pause = False

    def clear(self):
        self.clear_sentence()

    def draw_ui(self, frame, prediction, confidence, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 130), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        display_pred = self.display_map.get(prediction, prediction) if prediction else "--"
        cv2.putText(frame, f"Sign: {display_pred}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if confidence and confidence > 0:
            bar_x = 350
            bar_w = int(200 * confidence)
            bar_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 50), bar_color, -1)
            cv2.rectangle(frame, (bar_x, 30), (bar_x + 200, 50), (255, 255, 255), 1)
            cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + 210, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"Mode: {self.mode.upper()}", (w - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (w - 220, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if self._letter_buffer:
            buf_str = "".join(self._letter_buffer)
            elapsed = time.time() - self._last_gesture_time
            rem = max(0.0, self._word_end_pause - elapsed)
            cv2.putText(frame, f"Spelling: {buf_str} (pause in {rem:.1f}s)", (20, h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)

        sentence_str = " ".join(self.current_sentence) if self.current_sentence else "(empty)"
        cv2.putText(frame, f"Sentence: {sentence_str}", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        controls = "C: Clear | Space: Speak | F: Finalise | Q: Quit"
        cv2.putText(frame, controls, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        pTime = 0
        print(f"\n{'='*55}")
        print(f"  ISL 32-Class Recognition System ({self.mode.upper()} mode)")
        print(f"  Spellchecker: {'Enabled' if SPELLCHECK_AVAILABLE else 'Disabled'}")
        print(f"{'='*55}\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame, hands_detected = self.extract_landmarks(frame)

            current_pred = None
            current_conf = 0.0
            is_rest = not hands_detected or self.is_hand_at_rest(landmarks)
            hand_active = hands_detected and not is_rest
            if self.mode == 'static':
                if hand_active and not self._static_hand_active:
                    self._static_hand_settle_until = (
                        time.monotonic() + self._static_hand_settle_seconds
                    )
                self._static_hand_active = hand_active

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
                elif self.mode == 'static' and self._last_committed_letter:
                    self.neutral_frame_count = 0
                else:
                    self._handle_gesture_reset()

                if self.mode == 'static' and current_pred is not None and self.frame_count % 10 == 0:
                    top_indices = np.argsort(self._last_static_probabilities)[-3:][::-1]
                    print("[Live Diagnostic]")
                    for rank, index in enumerate(top_indices, start=1):
                        class_name = self.static_classes[index]
                        display_name = self.display_map.get(class_name, class_name)
                        confidence = float(self._last_static_probabilities[index]) * 100
                        print(f"Top {rank}: {display_name} = {confidence:.1f}%")

                    if stable_pred and (stable_pred.startswith('LETTER_') or stable_pred in self.alphabet_classes):
                        stable_letter = self.display_map.get(stable_pred, stable_pred.replace('LETTER_', ''))
                        print(f"Stable letter: {stable_letter}")
                    else:
                        print("Stable letter: none")
            else:
                if self.mode == 'dynamic' and len(self.sequence_buffer) > 0:
                    self.sequence_buffer.popleft()
                self.prediction_buffer.append(None)
                self._handle_gesture_reset()
                self._check_letter_pause()

            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if pTime > 0 else 0
            pTime = cTime

            out = self.draw_ui(annotated_frame, current_pred, current_conf, fps)
            cv2.imshow('ISL Recognition System', out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_sentence()
            elif key == ord('f'):
                self._finalise_letter_word()
            elif key == ord(' '):
                if self.current_sentence:
                    sentence = llm_translator.correct_grammar(self.current_sentence)
                    print(f"  Speaking: \"{sentence}\"")
                    self.tts.speak(sentence)

        self.landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time 32-Class ISL Recognition")
    parser.add_argument("--mode", type=str, default="static", choices=['static', 'dynamic', 'both'])
    parser.add_argument("--pause", type=float, default=None)
    args = parser.parse_args()

    recognizer = ISLRecognizer(mode=args.mode)
    if args.pause is not None:
        recognizer._word_end_pause = args.pause
    recognizer.run()
