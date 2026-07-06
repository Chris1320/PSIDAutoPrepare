import os
from pathlib import Path

NAME = "PSIDAutoPrepare"
VERSION = "0.1.0"

DEFAULT_TARGET_SIZE = (600, 600)

DEFAULT_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_FILENAME = "face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_FILEPATH = Path(os.getcwd()) / "models" / DEFAULT_MODEL_FILENAME
