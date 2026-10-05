    def clear(self):
        self.clear_sentence()

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