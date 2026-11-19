"""Load the with_mask / without_mask image folders into arrays."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from config import CLASSES, DATASET_DIR, IMAGE_SIZE, SEED, TEST_SPLIT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_images(dataset_dir: Path = DATASET_DIR, preprocess: bool = True):
    """Return (images, labels). Images are resized to 224x224."""
    images, labels = [], []
    for label in CLASSES:
        folder = dataset_dir / label
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No images found in {folder}. See README for dataset setup.")
        for path in files:
            image = img_to_array(load_img(path, target_size=IMAGE_SIZE))
            images.append(preprocess_input(image) if preprocess else image / 255.0)
            labels.append(label)
    return np.array(images, dtype="float32"), np.array(labels)


def encode_labels(labels):
    """One-hot encode labels with LabelBinarizer, as described in the paper."""
    binarizer = LabelBinarizer()
    encoded = binarizer.fit_transform(labels)
    if encoded.shape[1] == 1:  # LabelBinarizer gives a single column for 2 classes
        encoded = to_categorical(encoded, num_classes=2)
    return encoded, binarizer


def split(images, encoded_labels):
    """Stratified 80/20 train/test split."""
    return train_test_split(
        images, encoded_labels,
        test_size=TEST_SPLIT, stratify=encoded_labels.argmax(axis=1), random_state=SEED,
    )
