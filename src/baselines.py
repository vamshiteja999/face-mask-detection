"""Baseline models compared in the paper (Table 1 / Figure 11): SVM and a plain CNN.

The paper's CNN uses three pairs of 32-filter convolution layers, each pair followed by
max pooling, then dense layers for classification.

Usage:
    python src/baselines.py
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import layers, models

from config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, OUTPUT_DIR, SEED
from data import encode_labels, load_images, split

SVM_IMAGE_SIZE = (64, 64)  # smaller images keep the SVM tractable


def run_svm(x_train, x_test, y_train, y_test, class_names) -> str:
    def shrink(batch):
        import tensorflow as tf
        return tf.image.resize(batch, SVM_IMAGE_SIZE).numpy().reshape(len(batch), -1)

    svm = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=SEED),
                        SVC(kernel="rbf", C=10, gamma="scale"))
    svm.fit(shrink(x_train), y_train.argmax(axis=1))
    predictions = svm.predict(shrink(x_test))
    return classification_report(y_test.argmax(axis=1), predictions,
                                 target_names=class_names, digits=4)


def build_cnn() -> models.Model:
    model = models.Sequential([layers.Input(shape=(*IMAGE_SIZE, 3))])
    for _ in range(3):  # three pairs of 32-filter conv layers + max pooling
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def run_cnn(x_train, x_test, y_train, y_test, class_names) -> str:
    model = build_cnn()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE)
    predictions = model.predict(x_test).argmax(axis=1)
    return classification_report(y_test.argmax(axis=1), predictions,
                                 target_names=class_names, digits=4)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    images, labels = load_images(preprocess=False)  # scaled to [0, 1]
    encoded, binarizer = encode_labels(labels)
    x_train, x_test, y_train, y_test = split(images, encoded)

    results = {
        "SVM": run_svm(x_train, x_test, y_train, y_test, binarizer.classes_),
        "CNN": run_cnn(x_train, x_test, y_train, y_test, binarizer.classes_),
    }
    text = "\n".join(f"=== {name} ===\n{report}" for name, report in results.items())
    print(text)
    (OUTPUT_DIR / "baseline_reports.txt").write_text(text)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
