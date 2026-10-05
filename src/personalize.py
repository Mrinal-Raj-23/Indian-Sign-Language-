"""
Personalization data collector for User-Specific Fine-Tuning.
Reuses ISLRecognizer's extract_landmarks() and normalize_landmarks() pipeline
so collected samples are feature-compatible with the existing trained model.
"""

import os
import cv2
import numpy as np

try:
    from src import config
    from src.inference import ISLRecognizer
except ImportError:
    import config
    from inference import ISLRecognizer


class PersonalizationCollector:
    """
    Collects user-specific landmark samples for each sign class using the
    same MediaPipe + normalization pipeline as the base model.

    Usage (driven by the GUI):
        collector = PersonalizationCollector(recognizer, on_progress_cb)
        collector.start_class("Hello")
        # call collector.feed_frame(frame) each camera frame
        collector.finish_class()   # saves samples, advances to next class
        collector.save_dataset()   # writes final .npy files
    """

    def __init__(self, recognizer: ISLRecognizer, on_progress=None):
        """
        Args:
            recognizer:   The already-initialised ISLRecognizer (provides
                          extract_landmarks + normalize_landmarks).
            on_progress:  Optional callback(class_name, collected, total)
                          called after every accepted sample.
        """
        self.recognizer = recognizer
        self.on_progress = on_progress

        self.classes = list(getattr(config, 'PERSONAL_CLASSES',
                                    config.STATIC_CLASSES))
        self.samples_per_class = getattr(config, 'PERSONAL_SAMPLES_PER_CLASS', 25)
        self.save_dir = getattr(config, 'PERSONAL_DATA_PATH',
                                os.path.join(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__))),
                                    'data', 'personal'))
        os.makedirs(self.save_dir, exist_ok=True)

        # Per-session state
        self.current_class_idx = 0
        self.current_samples: list[np.ndarray] = []
        self.collecting = False

        # Accumulated dataset across all classes
        self._all_X: list[np.ndarray] = []
        self._all_y: list[str] = []

        # Load any previously saved session so the user can resume
        self._load_existing()

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def current_class(self) -> str:
        if self.current_class_idx < len(self.classes):
            return self.classes[self.current_class_idx]
        return ""

    @property
    def collected_count(self) -> int:
        return len(self.current_samples)

    @property
    def is_complete(self) -> bool:
        return self.current_class_idx >= len(self.classes)

    @property
    def progress_text(self) -> str:
        if self.is_complete:
            return "All classes complete!"
        return (f"{self.current_class}: "
                f"{self.collected_count}/{self.samples_per_class} samples")

    # ── Session control ──────────────────────────────────────────────────────

    def start_class(self, class_name: str | None = None):
        """Begin collecting samples for the current (or specified) class."""
        if class_name is not None and class_name in self.classes:
            self.current_class_idx = self.classes.index(class_name)
        self.current_samples = []
        self.collecting = True

    def stop_collecting(self):
        """Pause collection without advancing to the next class."""
        self.collecting = False

    def finish_class(self) -> bool:
        """
        Save collected samples for the current class and advance.
        Returns True if there are more classes remaining.
        """
        self.collecting = False
        if self.current_samples:
            cls = self.current_class
            arr = np.array(self.current_samples)
            self._all_X.extend(self.current_samples)
            self._all_y.extend([cls] * len(self.current_samples))
            self._save_class_file(cls, arr)

        self.current_class_idx += 1
        self.current_samples = []
        return not self.is_complete

    def skip_class(self):
        """Skip the current class without saving."""
        self.collecting = False
        self.current_class_idx += 1
        self.current_samples = []

    # ── Frame feeding ────────────────────────────────────────────────────────

    def feed_frame(self, frame) -> tuple[any, bool]:
        """
        Process one camera frame.  If collecting and a hand is detected,
        extract + normalize landmarks and store the sample.

        Args:
            frame: raw BGR frame from cv2.VideoCapture

        Returns:
            (annotated_frame, sample_accepted)
        """
        # ── Safeguard 10: guard against a bad frame crashing the inference loop ──
        try:
            annotated, _pred, _conf, hands_detected = self.recognizer.process_frame(frame)
        except (cv2.error, ValueError, RuntimeError) as frame_err:
            print(f"  [warn] process_frame error: {frame_err}")
            return frame, False

        sample_accepted = False
        if self.collecting and not self.is_complete:
            if hands_detected:
                # Re-extract raw landmarks from the last processed frame.
                # process_frame() already called extract_landmarks internally,
                # but we need the raw (un-normalised) array to normalise ourselves
                # identically to how training data was prepared.
                try:
                    landmarks, _, _ = self.recognizer.extract_landmarks(frame)
                except (cv2.error, ValueError, RuntimeError) as lm_err:
                    print(f"  [warn] extract_landmarks error: {lm_err}")
                    return annotated, False

                # ── Safeguard 5: validate feature dimensions before storing ───────
                if landmarks.shape != (126,):
                    print(f"  [warn] Unexpected landmark shape {landmarks.shape}; skipping.")
                    return annotated, False

                if not np.all(landmarks == 0):
                    normalized = self.recognizer.normalize_landmarks(landmarks)
                    self.current_samples.append(normalized)
                    sample_accepted = True

                    if self.on_progress:
                        self.on_progress(
                            self.current_class,
                            self.collected_count,
                            self.samples_per_class
                        )

                    # Auto-stop when target reached
                    if self.collected_count >= self.samples_per_class:
                        self.collecting = False

        return annotated, sample_accepted

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_class_file(self, class_name: str, arr: np.ndarray):
        """Save samples for one class to data/personal/<class_name>.npy"""
        # Sanitise class name to prevent path traversal
        safe_cls = os.path.basename(class_name)
        path = os.path.join(self.save_dir, f"{safe_cls}.npy")
        if os.path.exists(path):
            # ── Safeguard 8: corrupted existing file ──────────────────────────────
            try:
                existing = np.load(path, allow_pickle=False)
                arr = np.vstack([existing, arr])
            except (ValueError, OSError) as load_err:
                print(f"  [warn] Existing file for '{class_name}' is corrupted "
                      f"({load_err}); overwriting with new samples.")
        np.save(path, arr)

    def save_dataset(self):
        """
        Write the full session dataset as X.npy / y.npy in data/personal/.
        These files are what the fine-tuning step will load.
        """
        if not self._all_X:
            return
        X = np.array(self._all_X)
        y = np.array(self._all_y)
        np.save(os.path.join(self.save_dir, "X_personal.npy"), X)
        np.save(os.path.join(self.save_dir, "y_personal.npy"), y)

    def _load_existing(self):
        """Load any previously saved per-class files so the user can resume."""
        for cls in self.classes:
            safe_cls = os.path.basename(cls)
            path = os.path.join(self.save_dir, f"{safe_cls}.npy")
            if os.path.exists(path):
                # ── Safeguard 8: corrupted file on resume ─────────────────────────
                try:
                    arr = np.load(path, allow_pickle=False)
                    self._all_X.extend(arr.tolist())
                    self._all_y.extend([cls] * len(arr))
                except (ValueError, OSError) as load_err:
                    print(f"  [warn] Could not load existing data for '{cls}': {load_err}")

    def existing_count(self, class_name: str) -> int:
        """Return how many samples are already saved for a class."""
        safe_cls = os.path.basename(class_name)
        path = os.path.join(self.save_dir, f"{safe_cls}.npy")
        if os.path.exists(path):
            try:
                return len(np.load(path, allow_pickle=False))
            except (ValueError, OSError) as load_err:
                print(f"  [warn] Could not count samples for '{class_name}': {load_err}")
        return 0
