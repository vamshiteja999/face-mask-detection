"""Train the MobileNet-based face mask classifier (the paper's main model).

Usage:
    python src/train.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (BATCH_SIZE, EPOCHS, IMAGE_SIZE, LEARNING_RATE, MODEL_PATH, OUTPUT_DIR,
                    PLOT_PATH, REPORT_PATH)
from data import encode_labels, load_images, split


def build_model() -> Model:
    """MobileNetV2 backbone (ImageNet weights, frozen) + small classification head.

    The head uses average pooling, which the paper reports for a smoother feature map.
    """
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_tensor=Input(shape=(*IMAGE_SIZE, 3)))
    base.trainable = False

    x = AveragePooling2D(pool_size=(7, 7))(base.output)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="softmax")(x)
    return Model(inputs=base.input, outputs=outputs)


def plot_history(history) -> None:
    """Training/validation loss and accuracy per epoch (Figure 12 in the paper)."""
    plt.style.use("ggplot")
    plt.figure()
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        plt.plot(np.arange(1, EPOCHS + 1), history.history[key], label=key)
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("[INFO] loading images...")
    images, labels = load_images()
    encoded, binarizer = encode_labels(labels)
    x_train, x_test, y_train, y_test = split(images, encoded)
    print(f"[INFO] {len(x_train)} train / {len(x_test)} test images")

    augment = ImageDataGenerator(
        rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest",
    )

    model = build_model()
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    print("[INFO] training...")
    history = model.fit(
        augment.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
    )

    print("[INFO] evaluating...")
    predictions = model.predict(x_test, batch_size=BATCH_SIZE).argmax(axis=1)
    report = classification_report(y_test.argmax(axis=1), predictions,
                                   target_names=binarizer.classes_, digits=4)
    print(report)
    REPORT_PATH.write_text(report)

    model.save(MODEL_PATH)
    plot_history(history)
    print(f"[INFO] saved model to {MODEL_PATH}, plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
