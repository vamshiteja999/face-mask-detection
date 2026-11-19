"""Run mask detection on a single image.

Usage:
    python src/detect_image.py --image path/to/photo.jpg [--output result.jpg]
"""
import argparse

import cv2

from detector import MaskDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to the input image")
    parser.add_argument("--output", help="optional path to save the annotated image")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)

    result = MaskDetector().annotate(frame)
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"[INFO] saved {args.output}")
    else:
        cv2.imshow("Face Mask Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
