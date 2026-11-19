"""Shared settings, matching the setup described in the ICACECS 2022 paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"          # dataset/with_mask, dataset/without_mask
OUTPUT_DIR = ROOT / "outputs"

CLASSES = ["with_mask", "without_mask"]  # 2 classes, as in the paper
IMAGE_SIZE = (224, 224)                  # MobileNet input size used in the paper
TEST_SPLIT = 0.20                        # 80% train / 20% test
EPOCHS = 20                              # paper: 20 epochs
BATCH_SIZE = 32                          # paper: batch size 32
LEARNING_RATE = 1e-4
SEED = 42

MODEL_PATH = OUTPUT_DIR / "mask_detector.keras"
PLOT_PATH = OUTPUT_DIR / "training_plot.png"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
