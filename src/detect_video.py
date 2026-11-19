"""Live mask detection from a webcam (or a video file). Press 'q' to quit.

Usage:
    python src/detect_video.py              # default webcam
    python src/detect_video.py --source clip.mp4
"""
import argparse
import time

import cv2

from detector import MaskDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="webcam index or video file path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    detector = MaskDetector()
    previous = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = detector.annotate(frame)

        now = time.time()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
