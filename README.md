# Face Mask Detection using TensorFlow and OpenCV

Real-time face mask detection: a Haar cascade finds faces, and a MobileNet-based classifier labels each face **Mask** (green box) or **No Mask** (red box) with a confidence score, on images or live webcam video.

This repository is a clean implementation of the method from our paper:

> Jeevitha R, Dr. Nirmala M, **Kaspe Vamshi Teja**, K Charish SS Krishna, Sai Krishna Srinivas.
> *Detection of Face Mask using AI/TensorFlow.* ICACECS 2022 (paper #146).
> New Horizon College of Engineering, Bengaluru, India.

The code was reorganized and rewritten in 2026 from the paper's method for this portfolio.

## How it works

```
Image / webcam frame
      │
      ▼
Haar cascade face detection (OpenCV)
      │  crop each face
      ▼
Resize to 224×224 → MobileNetV2 preprocessing
      │
      ▼
MobileNetV2 backbone (ImageNet weights, frozen)
  → AveragePooling(7×7) → Dense(128) → Dropout(0.5) → Softmax(2)
      │
      ▼
"Mask" (green box) / "No Mask" (red box) + confidence %
```

| Setting | Value (from the paper) |
|---|---|
| Classes | `with_mask`, `without_mask` |
| Dataset | 1,376 images (690 with mask, 686 without mask) |
| Split | 80% train / 20% test |
| Input size | 224 × 224 |
| Labels | One-hot encoded with `LabelBinarizer` |
| Training | 20 epochs, batch size 32, data augmentation |
| Baselines | SVM, plain CNN (3 pairs of 32-filter conv layers + max pooling) |

## Results

### This implementation

MobileNetV2 classifier, 1,376-image dataset, 80/20 stratified split (276 test images), 20 epochs, trained on a MacBook Air (Apple M1, 8 GB):

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| with_mask | 1.0000 | 0.9928 | 0.9964 | 138 |
| without_mask | 0.9928 | 1.0000 | 0.9964 | 138 |
| **Overall accuracy** | | | **99.64%** | 276 |

One test image out of 276 was misclassified.

![Training loss and accuracy](docs/training_plot.png)

![Live detection demo](docs/demo.png)

**Why the score is so high, and why it won't hold in the real world.** Many "with mask" images in this dataset were created by overlaying a mask onto "without mask" photos using facial landmarks. The same faces can therefore appear in both training and test sets, and the synthetic masks all look alike, which makes the test easier than real camera footage. A person-level split or a test set of real masked photos would give a more honest estimate.

### Results reported in the paper

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| SVM | 92.01 | 85 | 89 | 87 |
| CNN | 91.03 | 86 | 88 | 86 |
| **MobileNet (depthwise-separable)** | **93.16** | **90** | **91** | **89** |

These are the averages from the paper's comparison table. Run `python src/baselines.py` to reproduce the SVM and CNN baselines on the same split.

## Project structure

```
src/
  config.py         # paths and hyperparameters
  data.py           # image loading, label encoding, 80/20 split
  train.py          # MobileNetV2 classifier training + evaluation + training plot
  baselines.py      # SVM and CNN baselines from the paper
  detector.py       # Haar cascade + classifier, draws green/red boxes
  detect_image.py   # run on one image
  detect_video.py   # run on webcam or video file (shows FPS)
dataset/
  with_mask/        # put "with mask" images here
  without_mask/     # put "without mask" images here
outputs/            # trained model, plots and reports (created by training)
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dataset:** place images in `dataset/with_mask/` and `dataset/without_mask/`. The paper used the public 1,376-image with/without-mask dataset (690 + 686 images) by Prajna Bhandary. Images are not committed to this repo.

```bash
python src/train.py                                  # train MobileNetV2 classifier
python src/baselines.py                              # optional: SVM and CNN baselines
python src/detect_image.py --image photo.jpg --output result.jpg
python src/detect_video.py                           # live webcam, press q to quit
```

## Limitations and next steps

- Haar cascades miss faces that are turned sideways, far away, or poorly lit. A DNN face detector (for example OpenCV's SSD face detector) would be more robust.
- The dataset is small and many "with mask" images are synthetic (a mask overlaid using facial landmarks), so real-world accuracy will be lower than the reported numbers.
- Future work from the paper: alarms when many unmasked people are present, and gate or door control at entrances.

## License

Copyright © 2022–2026 the authors. All rights reserved. Please contact me before reusing this code.
