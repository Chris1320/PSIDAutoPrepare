import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from psidautoprepare.info import (
    DEFAULT_MODEL_FILEPATH,
    DEFAULT_MODEL_URL,
    DEFAULT_TARGET_SIZE,
)


def get_face_detector() -> cv2.FaceDetectorYN:
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
        input_size=DEFAULT_TARGET_SIZE,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    return detector


def save_with_dpi(cv2_image: cv2.typing.MatLike, output_path: Path):
    """Converts OpenCV image to Pillow and saves with strict DPI metadata."""
    # OpenCV uses BGR, but Pillow expects RGB. We must convert the color space first.
    rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # Save the file with the DPI embedded in the EXIF/header data
    pil_img.save(
        str(output_path),
        format="JPEG",
        quality=100,
        dpi=DEFAULT_TARGET_SIZE,
        subsampling=0,
        progressive=True,
    )


def main(target_dir: str, output_dir: str | None, err_output: str) -> int:
    input_path = Path(target_dir)

    if not input_path.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist or is not a directory.")
        return 1

    output_path = (
        Path(output_dir) if output_dir is not None else Path(input_path) / "output"
    )
    output_path.mkdir(exist_ok=True)
    print(f"Output directory set to: {output_path}")

    detector = get_face_detector()

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not files:
        print(f"No valid images found in {input_path}")
        return 0

    print(f"Found {len(files)} images to process.\n---")
    failed_detections: list[str] = []

    for img_file in tqdm(files, desc="Processing images"):
        out_file = output_path / img_file.name

        img = cv2.imread(str(img_file))
        if img is None:
            tqdm.write(f"[-] Could not read {img_file.name}. Skipping.")
            continue

        height, width, _ = img.shape
        detector.setInputSize((width, height))
        _, faces = detector.detect(img)

        if faces is not None and len(faces) > 0:  # type: ignore
            fx, fy, fw, fh = map(int, faces[0][:4])
            padding = int(fh * 0.6)

            crop_x1 = max(0, fx - padding)
            crop_y1 = max(0, fy - padding)
            crop_x2 = min(width, fx + fw + padding)
            crop_y2 = min(height, fy + fh + padding)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            square_size = min(crop_w, crop_h)

            crop_x1 = crop_x1 + (crop_w - square_size) // 2
            crop_y1 = crop_y1 + (crop_h - square_size) // 2
            crop_x2 = crop_x1 + square_size
            crop_y2 = crop_y1 + square_size

            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # Interpolation set to LANCZOS4 which is better for upscaling if needed
            final_img = cv2.resize(
                cropped_img, DEFAULT_TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4
            )

            # Use our new Pillow save function
            save_with_dpi(final_img, out_file)
            tqdm.write(f"[+] Successfully cropped: {img_file.name}")

        else:
            tqdm.write(
                f"[!] No face detected in {img_file.name}. Applying a center crop fallback."
            )
            failed_detections.append(img_file.name)

            min_dim = min(width, height)
            cx1 = (width - min_dim) // 2
            cy1 = (height - min_dim) // 2

            fallback_img = img[cy1 : cy1 + min_dim, cx1 : cx1 + min_dim]
            final_img = cv2.resize(
                fallback_img, DEFAULT_TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4
            )

            # Use our new Pillow save function
            save_with_dpi(final_img, out_file)

    print("\nProcessing complete!")

    if failed_detections:
        err_file_path = input_path / err_output
        with open(err_file_path, "w", encoding="utf-8") as f:
            for name in failed_detections:
                f.write(f"{name}\n")
        print(
            f"Logged {len(failed_detections)} images with no face detected to: {err_file_path}"
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop student pictures into 2x2 IDs with strict DPI metadata."
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="Target directory containing the raw student images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory where cropped images will be saved.",
        default=None,
    )
    parser.add_argument(
        "--err-output",
        type=str,
        default="no_face_detected.txt",
        help="Filename to log images where no face was detected (default: no_face_detected.txt).",
    )
    args = parser.parse_args()

    sys.exit(main(args.target_dir, args.output_dir, args.err_output))
