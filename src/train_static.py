import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from src import config
except ImportError:
    import config

STATIC_MODEL_FILENAME = 'static_model_32class.h5'
LABEL_ENCODER_FILENAME = 'label_encoder_32class.npy'

def build_model(num_features, num_classes):
    """
    Builds a Dense Neural Network for static gesture recognition.
    Output layer has exactly num_classes units.
    """
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def train_model(epochs=None, batch_size=None):
    if epochs is None:
        epochs = getattr(config, 'EPOCHS', 50)
    if batch_size is None:
        batch_size = getattr(config, 'BATCH_SIZE', 32)

    proc_path = config.PROCESSED_DATA_PATH
    models_path = config.MODELS_PATH
    os.makedirs(models_path, exist_ok=True)

    print(f"Loading preprocessed data from {proc_path}...")
    try:
        X_train = np.load(os.path.join(proc_path, 'X_train.npy'))
        X_test  = np.load(os.path.join(proc_path, 'X_test.npy'))
        y_train = np.load(os.path.join(proc_path, 'y_train.npy'))
        y_test  = np.load(os.path.join(proc_path, 'y_test.npy'))
        classes = np.load(os.path.join(proc_path, 'label_encoder_32class.npy'), allow_pickle=True)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        print("Please run preprocess.py after collecting the 32 classes.")
        return None, None

    num_classes  = len(classes)
    num_features = X_train.shape[1]

    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Total classes to train: {num_classes}")
    if num_classes != 32:
        print(f"WARNING: Expected exactly 32 classes, found {num_classes} classes!")
    else:
        print("Verified: Exactly 32 classes present.")

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    model = build_model(num_features, num_classes)
    model_save_path = os.path.join(models_path, STATIC_MODEL_FILENAME)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print(f"Training 32-class Dense Neural Network for {epochs} epochs...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('32-Class Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('32-Class Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    history_path = os.path.join(models_path, 'training_history_32class.png')
    plt.savefig(history_path)
    plt.close()

    # Evaluation
    print("\nEvaluating 32-class model on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - 32 Class Static ISL Recognizer')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(models_path, 'confusion_matrix_32class.png')
    plt.savefig(cm_path)
    plt.close()

    print(f"\nModel saved successfully to: {model_save_path}")
    print(f"Label encoder confirmed at:  {os.path.join(proc_path, LABEL_ENCODER_FILENAME)}")
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 32-class static gesture model.")
    parser.add_argument('--epochs', type=int, default=getattr(config, 'EPOCHS', 50), help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=getattr(config, 'BATCH_SIZE', 32), help='Batch size')
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size)
