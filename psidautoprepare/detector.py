import urllib.request

import cv2

from psidautoprepare.info import (
    DEFAULT_MODEL_FILEPATH,
    DEFAULT_MODEL_URL,
    DEFAULT_TARGET_SIZE,
)


def get_face_detector(
    input_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> cv2.FaceDetectorYN:
    """
    Downloads the YuNet model if missing and initializes the detector.

    Returns:
        An initialized face detector.
    """

    model_path = DEFAULT_MODEL_FILEPATH

    if not model_path.exists():
        print(f"[*] Downloading YuNet model to {model_path}...")
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
        print("[*] Download complete.")

    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=input_size,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    return detector
