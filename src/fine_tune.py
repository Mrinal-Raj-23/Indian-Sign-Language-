"""
User-Specific Fine-Tuning for ISL static gesture recognition.

Architecture (static_model_6class.h5):
  [0] dense                Dense(256, relu)       ─┐
  [1] batch_normalization  BN                      │  FROZEN  (feature extractor)
  [2] dropout              Dropout(0.3)            │
  [3] dense_1              Dense(128, relu)        │
  [4] batch_normalization_1 BN                    ─┘
  [5] dropout_1            Dropout(0.3)           ─┐
  [6] dense_2              Dense(64, relu)         │  TRAINABLE (fine-tune)
  [7] batch_normalization_2 BN                     │
  [8] dense_3              Dense(7, softmax)      ─┘

Input : (None, 126)   — wrist-relative, scale-invariant normalised landmarks
Output: (None, 7)     — ['Bye','Hello','I','Namaste','Sorry','Thank You','You']
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# Prevent matplotlib DLL issues on Windows
for _mod in ('matplotlib', 'matplotlib.pyplot', 'matplotlib.ft2font'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

try:
    from src import config
except ImportError:
    import config

# ── Constants derived from inspection ────────────────────────────────────────
# Layers 0-4 are frozen (Dense-256 block + Dense-128 block).
# Layers 5-8 are fine-tuned (Dropout-1, Dense-64, BN-2, Dense-7).
_FREEZE_UP_TO_IDX = 4          # inclusive — layers 0,1,2,3,4 frozen
_MIN_SAMPLES_PER_CLASS = 5     # refuse to train below this
_FINE_TUNE_LR = 1e-4           # small LR to avoid destroying base weights
_FINE_TUNE_EPOCHS = 40
_BATCH_SIZE = 8                # small batch suits tiny datasets


# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_FEATURES = 126   # 21 landmarks × 3 coords × 2 hands


def _load_personal_data(personal_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load per-class .npy files from data/personal/.
    Returns (X, y_str, class_names) where y_str contains string labels.
    Raises ValueError if data is missing, too sparse, or has wrong feature shape.
    """
    # ── Safeguard 1: missing dataset directory ────────────────────────────────
    if not os.path.exists(personal_dir):
        raise ValueError(
            f"Personal data directory not found: {personal_dir}. "
            "Collect data first using the Personalize panel."
        )

    X_parts, y_parts = [], []
    class_names = []

    personal_classes = getattr(config, 'PERSONAL_CLASSES', config.STATIC_CLASSES)

    for cls in personal_classes:
        # Sanitise class name to prevent path traversal (safeguard: path safety)
        safe_cls = os.path.basename(cls)
        path = os.path.join(personal_dir, f"{safe_cls}.npy")
        if not os.path.exists(path):
            print(f"  [skip] No data for class '{cls}'")
            continue

        # ── Safeguard 8: corrupted saved file ─────────────────────────────────
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as load_err:
            print(f"  [skip] '{cls}' — file corrupted or unreadable: {load_err}")
            continue

        # ── Safeguard 2: insufficient samples ─────────────────────────────────
        if len(arr) < _MIN_SAMPLES_PER_CLASS:
            print(f"  [skip] '{cls}' has only {len(arr)} samples "
                  f"(minimum {_MIN_SAMPLES_PER_CLASS})")
            continue

        # ── Safeguard 5: incorrect feature dimensions ─────────────────────────
        if arr.ndim != 2 or arr.shape[1] != _EXPECTED_FEATURES:
            print(f"  [skip] '{cls}' has wrong shape {arr.shape} "
                  f"(expected (N, {_EXPECTED_FEATURES}))")
            continue

        X_parts.append(arr)
        y_parts.extend([cls] * len(arr))
        class_names.append(cls)
        print(f"  [load] '{cls}': {len(arr)} samples")

    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {_MIN_SAMPLES_PER_CLASS} samples each. "
            f"Found: {class_names}. Collect more data first."
        )

    X = np.vstack(X_parts)
    y = np.array(y_parts)
    return X, y, class_names


def _build_fine_tune_model(base_model_path: str,
                           num_classes: int,
                           class_names: list[str]) -> tf.keras.Model:
    """
    Load the base model, freeze layers 0-4, keep layers 5-8 trainable.
    Replace the output layer only if the number of classes differs.
    Returns the compiled model ready for fine-tuning.
    Raises ValueError on corrupted model or incompatible architecture.
    """
    # ── Safeguard 8: corrupted base model ─────────────────────────────────────
    try:
        base = tf.keras.models.load_model(base_model_path)
    except Exception as load_err:
        raise ValueError(f"Base model is corrupted or incompatible: {load_err}") from load_err

    # ── Safeguard 5: validate input feature dimension ─────────────────────────
    expected_input = (_EXPECTED_FEATURES,)
    actual_input = base.input_shape[1:]
    if actual_input != expected_input:
        raise ValueError(
            f"Base model expects input shape {expected_input}, "
            f"but personal data has {_EXPECTED_FEATURES} features. "
            "Re-collect data or retrain the base model."
        )

    # ── Freeze feature-extractor layers (0 through _FREEZE_UP_TO_IDX) ────────
    for layer in base.layers[:_FREEZE_UP_TO_IDX + 1]:
        layer.trainable = False

    # ── Ensure fine-tune layers are trainable ─────────────────────────────────
    for layer in base.layers[_FREEZE_UP_TO_IDX + 1:]:
        layer.trainable = True

    base_num_classes = base.output_shape[-1]

    if num_classes == base_num_classes:
        # Class count matches — reuse the output layer as-is
        model = base
    else:
        # ── Safeguard 6: class count differs — replace output layer ───────────
        print(f"  Replacing output layer: {base_num_classes} → {num_classes} classes")
        # Ensure the model has enough layers to safely index [-2]
        if len(base.layers) < 2:
            raise ValueError(
                f"Base model has only {len(base.layers)} layer(s); "
                "cannot replace output layer."
            )
        x = base.layers[6].output           # dense_2 (64-unit layer, index 6)
        out = tf.keras.layers.Dense(
            num_classes, activation='softmax', name='dense_personal'
        )(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=_FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _print_layer_status(model: tf.keras.Model):
    """Print which layers are frozen vs trainable."""
    print("\n  Layer freeze status:")
    for i, layer in enumerate(model.layers):
        status = "TRAINABLE" if layer.trainable else "frozen   "
        tw = len(layer.trainable_weights)
        print(f"    [{i}] {status}  {layer.name:35s} tw={tw}")


def fine_tune(
    progress_callback=None,
    status_callback=None
) -> dict:
    """
    Main entry point — loads personal data, fine-tunes the base model,
    saves static_model_personal.h5.

    Args:
        progress_callback: optional callable(epoch, total_epochs, logs)
                           for GUI progress updates.

    Returns:
        dict with keys: 'success', 'message', 'model_path',
                        'final_accuracy', 'val_accuracy', 'epochs_run'
    """
    models_path = getattr(config, 'MODELS_PATH', str(getattr(config, 'MODELS_DIR', 'models')))
    personal_dir = getattr(config, 'PERSONAL_DATA_PATH',
                           os.path.join(os.path.dirname(
                               os.path.dirname(os.path.abspath(__file__))),
                               'data', 'personal'))
    base_model_path = os.path.join(models_path, 'static_model_6class.h5')
    personal_model_path = os.path.join(models_path, 'static_model_personal.h5')

    def _status(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    print("\n" + "=" * 55)
    print("  ISL User-Specific Fine-Tuning")
    print("=" * 55)

    # ── 1. Validate base model exists ─────────────────────────────────────────
    if not os.path.exists(base_model_path):
        msg = f"Base model not found: {base_model_path}"
        print(f"  ERROR: {msg}")
        return {'success': False, 'message': msg}

    # ── 2. Load personal data ─────────────────────────────────────────────────
    _status("Preparing data...")
    try:
        X, y_str, class_names = _load_personal_data(personal_dir)
    except ValueError as e:
        return {'success': False, 'message': str(e)}

    print(f"\n  Dataset: {len(X)} samples, {len(class_names)} classes")
    print(f"  Classes: {class_names}")
    print(f"  Feature shape: {X.shape[1:]}")

    # ── 3. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(class_names)
    y_enc = le.transform(y_str)
    num_classes = len(class_names)
    y_cat = to_categorical(y_enc, num_classes)

    # ── 4. Train/val split — stratified, handles tiny datasets ───────────────
    # With very few samples per class we use a fixed val fraction but
    # guarantee at least 1 sample per class in validation.
    samples_per_class = {cls: int(np.sum(y_str == cls)) for cls in class_names}
    min_samples = min(samples_per_class.values())

    if min_samples >= 10:
        val_split = 0.2
    elif min_samples >= 6:
        val_split = 0.15
    else:
        val_split = 0.0   # too few — train on everything, no val split

    if val_split > 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=val_split,
            random_state=42, stratify=y_enc
        )
        validation_data = (X_val, y_val)
        print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  (val_split={val_split})")
    else:
        X_train, y_train = X, y_cat
        validation_data = None
        print(f"\n  Train: {len(X_train)}  Val: none (too few samples for split)")

    # ── 5. Build fine-tune model ──────────────────────────────────────────────
    _status("Fine-tuning model...")
    print("\n[2/4] Building fine-tune model...")
    model = _build_fine_tune_model(base_model_path, num_classes, class_names)
    _print_layer_status(model)

    trainable_params = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    total_params = model.count_params()
    print(f"\n  Trainable params : {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    # ── 6. Callbacks ──────────────────────────────────────────────────────────
    callbacks = []

    if validation_data is not None:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss', patience=10,
                restore_best_weights=True, verbose=1
            )
        )
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6, verbose=1
            )
        )
    else:
        # No val data — stop on training loss plateau
        callbacks.append(
            EarlyStopping(
                monitor='loss', patience=12,
                restore_best_weights=True, verbose=1
            )
        )

    # Optional GUI progress callback
    if progress_callback is not None:
        class _ProgressCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_callback(epoch + 1, _FINE_TUNE_EPOCHS, logs or {})
        callbacks.append(_ProgressCB())

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n[3/4] Fine-tuning for up to {_FINE_TUNE_EPOCHS} epochs "
          f"(lr={_FINE_TUNE_LR}, batch={_BATCH_SIZE})...")
    _status(f"Training... (up to {_FINE_TUNE_EPOCHS} epochs)")

    # ── Safeguard 9: training failure ─────────────────────────────────────────
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=_FINE_TUNE_EPOCHS,
            batch_size=_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as train_err:
        msg = f"Training failed: {train_err}"
        print(f"  ERROR: {msg}")
        _status(f"❌ {msg}")
        return {'success': False, 'message': msg}

    epochs_run = len(history.history['accuracy'])
    final_train_acc = float(history.history['accuracy'][-1])
    final_val_acc = (float(history.history['val_accuracy'][-1])
                     if 'val_accuracy' in history.history else None)

    print(f"\n  Epochs run      : {epochs_run}")
    print(f"  Train accuracy  : {final_train_acc * 100:.1f}%")
    if final_val_acc is not None:
        print(f"  Val accuracy    : {final_val_acc * 100:.1f}%")

    # ── 8. Save — never overwrites base model ─────────────────────────────────
    _status("Training completed.")
    print(f"\n[4/4] Saving personalised model...")
    try:
        model.save(personal_model_path)
    except Exception as save_err:
        msg = f"Failed to save model: {save_err}"
        print(f"  ERROR: {msg}")
        return {'success': False, 'message': msg}

    # Save the label encoder so inference knows the class order
    le_path = os.path.join(personal_dir, 'label_encoder_personal.npy')
    np.save(le_path, le.classes_)

    _status("Personalized model saved.")
    print(f"  Model saved : {personal_model_path}")
    print(f"  Encoder saved: {le_path}")
    print("=" * 55 + "\n")

    return {
        'success': True,
        'message': 'Fine-tuning complete.',
        'model_path': personal_model_path,
        'final_accuracy': final_train_acc,
        'val_accuracy': final_val_acc,
        'epochs_run': epochs_run,
        'classes': class_names,
    }


def evaluate(
    status_callback=None
) -> dict:
    """
    Evaluate both the base model and the personalized model on a held-out
    test split carved from the user's collected personal data.

    The split uses the same random_state=42 and the same val_split logic as
    fine_tune(), so the test samples were never seen during training.

    Returns dict with keys:
        'success', 'message',
        'base_accuracy', 'personal_accuracy', 'improvement',
        'n_test', 'classes'
    """
    from sklearn.model_selection import train_test_split

    models_path = getattr(config, 'MODELS_PATH',
                          str(getattr(config, 'MODELS_DIR', 'models')))
    personal_dir = getattr(config, 'PERSONAL_DATA_PATH',
                           os.path.join(os.path.dirname(
                               os.path.dirname(os.path.abspath(__file__))),
                               'data', 'personal'))
    base_model_path    = os.path.join(models_path, 'static_model_6class.h5')
    personal_model_path = os.path.join(models_path, 'static_model_personal.h5')
    personal_le_path   = os.path.join(personal_dir, 'label_encoder_personal.npy')

    def _status(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    print("\n" + "=" * 55)
    print("  ISL Personalization Evaluation")
    print("=" * 55)

    # ── 1. Guard: both models must exist ─────────────────────────────────────
    if not os.path.exists(base_model_path):
        return {'success': False,
                'message': f'Base model not found: {base_model_path}'}
    if not os.path.exists(personal_model_path):
        return {'success': False,
                'message': 'Personalized model not found. Train it first.'}
    if not os.path.exists(personal_le_path):
        return {'success': False,
                'message': 'Personal label encoder not found. Re-train the model.'}

    # ── 2. Load personal data ─────────────────────────────────────────────────
    _status("Loading personal data...")
    try:
        X, y_str, class_names = _load_personal_data(personal_dir)
    except ValueError as e:
        return {'success': False, 'message': str(e)}

    # ── 3. Encode labels (same order as fine_tune) ────────────────────────────
    le = LabelEncoder()
    le.fit(class_names)
    y_enc = le.transform(y_str)
    num_classes = len(class_names)

    # ── 4. Carve out the SAME held-out split used during training ─────────────
    # Mirror the val_split logic from fine_tune() exactly.
    min_samples = min(int(np.sum(y_str == c)) for c in class_names)
    if min_samples >= 10:
        val_split = 0.2
    elif min_samples >= 6:
        val_split = 0.15
    else:
        # Dataset too small to split — evaluate on all data and warn
        val_split = 0.0

    if val_split > 0:
        _, X_test, _, y_test_enc = train_test_split(
            X, y_enc,
            test_size=val_split,
            random_state=42,       # identical seed to fine_tune()
            stratify=y_enc
        )
    else:
        # No split was done during training either, so use all data
        # (training accuracy is the best proxy available)
        X_test, y_test_enc = X, y_enc
        print("  Warning: dataset too small to split — "
              "evaluating on all samples (includes training data).")

    n_test = len(X_test)
    print(f"  Test samples : {n_test}  |  Classes: {class_names}")

    # ── 5. Load personal label encoder to map predictions correctly ───────────
    # ── Safeguard 8: corrupted label encoder ──────────────────────────────────
    try:
        personal_classes = np.load(personal_le_path, allow_pickle=True).tolist()
    except Exception as le_err:
        return {'success': False,
                'message': f'Personal label encoder is corrupted: {le_err}. Re-train the model.'}

    # ── Safeguard 7: class-label mismatch between data and encoder ────────────
    missing_in_encoder = [c for c in class_names if c not in personal_classes]
    if missing_in_encoder:
        return {'success': False,
                'message': (
                    f'Class-label mismatch: {missing_in_encoder} are in the data '
                    'but not in the saved label encoder. Re-train the model.'
                )}

    # ── 6. Evaluate base model ────────────────────────────────────────────────
    _status("Evaluating base model...")
    # ── Safeguard 8: corrupted base model ─────────────────────────────────────
    try:
        base_model = tf.keras.models.load_model(base_model_path)
    except Exception as bm_err:
        return {'success': False, 'message': f'Base model is corrupted: {bm_err}'}

    # ── Safeguard 5: validate feature dimensions before prediction ────────────
    if X_test.shape[1] != _EXPECTED_FEATURES:
        return {'success': False,
                'message': (
                    f'Feature dimension mismatch: test data has {X_test.shape[1]} features, '
                    f'expected {_EXPECTED_FEATURES}. Re-collect personal data.'
                )}

    base_preds = np.argmax(base_model.predict(X_test, verbose=0), axis=1)
    # base model classes come from the base label encoder
    base_le_path = os.path.join(
        getattr(config, 'PROCESSED_DATA_PATH',
                str(getattr(config, 'DATA_PROCESSED_DIR', 'data/processed'))),
        'label_encoder_6class.npy'
    )
    # ── Safeguard 3: missing base label encoder ────────────────────────────────
    if not os.path.exists(base_le_path):
        return {'success': False,
                'message': f'Base label encoder not found: {base_le_path}'}
    try:
        base_classes = np.load(base_le_path, allow_pickle=True).tolist()
    except Exception as ble_err:
        return {'success': False,
                'message': f'Base label encoder is corrupted: {ble_err}'}
    # Map base model integer predictions → class names → personal encoder ints
    # ── Safeguard 7: guard index out-of-range on base_classes lookup ──────────
    base_pred_names = [
        base_classes[i] if i < len(base_classes) else '__unknown__'
        for i in base_preds
    ]
    # Only count predictions for classes the personal encoder knows
    valid_mask = np.array([n in personal_classes for n in base_pred_names])
    base_pred_enc = np.array(
        [personal_classes.index(n) if n in personal_classes else -1
         for n in base_pred_names]
    )
    # Re-encode y_test using personal_classes order
    y_test_personal = np.array(
        [personal_classes.index(class_names[i])
         if i < len(class_names) and class_names[i] in personal_classes else -1
         for i in y_test_enc]
    )
    base_correct = int(np.sum(
        (base_pred_enc == y_test_personal) & valid_mask
    ))
    base_acc = base_correct / n_test
    print(f"  Base model   : {base_correct}/{n_test} correct  ({base_acc*100:.1f}%)")

    # ── 7. Evaluate personalized model ───────────────────────────────────────
    _status("Evaluating personalized model...")
    # ── Safeguard 8: corrupted personalized model ─────────────────────────────
    try:
        personal_model = tf.keras.models.load_model(personal_model_path)
    except Exception as pm_err:
        return {'success': False,
                'message': f'Personalized model is corrupted: {pm_err}. Re-train it.'}

    pers_preds = np.argmax(personal_model.predict(X_test, verbose=0), axis=1)
    # ── Safeguard 6: personal model output class count must match encoder ──────
    pers_output_classes = personal_model.output_shape[-1]
    if pers_output_classes != len(personal_classes):
        return {'success': False,
                'message': (
                    f'Personalized model outputs {pers_output_classes} classes '
                    f'but label encoder has {len(personal_classes)}. Re-train the model.'
                )}
    # personal model output indices map directly to personal_classes
    pers_correct = int(np.sum(pers_preds == y_test_personal))
    pers_acc = pers_correct / n_test
    print(f"  Personal model: {pers_correct}/{n_test} correct  ({pers_acc*100:.1f}%)")

    improvement = pers_acc - base_acc
    print(f"  Improvement  : {improvement*100:+.1f}%")
    print("=" * 55 + "\n")

    return {
        'success': True,
        'message': 'Evaluation complete.',
        'base_accuracy': base_acc,
        'personal_accuracy': pers_acc,
        'improvement': improvement,
        'n_test': n_test,
        'classes': class_names,
    }


if __name__ == '__main__':
    result = fine_tune()
    if not result['success']:
        print(f"Failed: {result['message']}")
        sys.exit(1)
