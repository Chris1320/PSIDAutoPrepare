import os
from pathlib import Path

NAME = "PSIDAutoPrepare"
VERSION = "0.1.0"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_TARGET_SIZE = (2048, 2048)
DEFAULT_DPI = 300
DEFAULT_PADDING_SIZE = 150

DEFAULT_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_FILENAME = "face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_FILEPATH = Path(os.getcwd()) / "models" / DEFAULT_MODEL_FILENAME
