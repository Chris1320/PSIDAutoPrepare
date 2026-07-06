import argparse
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from psidautoprepare.info import (
    DEFAULT_MODEL_FILEPATH,
    DEFAULT_MODEL_URL,
    DEFAULT_TARGET_SIZE,
    IMAGE_EXTENSIONS,
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


def normalize_target_brightness(
    cv2_image: cv2.typing.MatLike, target_percentage: int
) -> cv2.typing.MatLike:
    """
    Uses Gamma Correction to non-linearly adjust the midtones of the image
    so the overall average brightness matches the target percentage.
    This mimics the 'Midtone' slider in Photoshop's Levels adjustment.
    """
    # 1. Convert to LAB color space to isolate pure Lightness (L) from Color (A/B)
    lab = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Calculate current average lightness (0 to 255)
    current_mean = cv2.mean(l_channel)[0]

    # Safety check: if image is entirely pitch black or blindingly white, skip math
    if current_mean <= 1 or current_mean >= 254:
        return cv2_image

    # 3. Calculate target mean (0 to 255)
    target_mean = (target_percentage / 100.0) * 255.0

    # 4. Calculate the specific Gamma curve needed to bend the current mean to the target mean
    # Formula: target = current ^ gamma  =>  gamma = log(target) / log(current) (scaled 0-1)
    gamma = math.log(target_mean / 255.0) / math.log(current_mean / 255.0)

    # 5. Build a Look-Up Table (LUT) for blazing-fast mapping
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )

    # 6. Apply the non-linear curve strictly to the Lightness channel
    l_adjusted = cv2.LUT(l_channel, lut)

    # 7. Merge the adjusted lightness back with the original colors
    lab_adjusted = cv2.merge((l_adjusted, a_channel, b_channel))
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)


def save_with_dpi(cv2_image: cv2.typing.MatLike, output_path: Path):
    """Converts OpenCV image to Pillow and saves with strict 300 DPI metadata."""
    rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # Hardcode standard photo-print DPI (300x300) independent of pixel resolution
    pil_img.save(
        str(output_path),
        format="JPEG",
        quality=100,
        dpi=(300, 300),
        subsampling=0,
        progressive=True,
    )


def main(
    target_dir: str,
    output_dir: str | None,
    err_output: str,
    size: int,
    padding_ratio: float,
    exclude: str | None,
    normalize: int | None,
) -> int:
    input_path = Path(target_dir)
    target_size = (size, size)

    if not input_path.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist or is not a directory.")
        return 1

    output_path = (
        Path(output_dir) if output_dir is not None else Path(input_path) / "output"
    )
    output_path.mkdir(exist_ok=True)
    print(f"Output directory set to: {output_path}")
    print(f"Target Resolution: {size}x{size}px | Padding Ratio: {padding_ratio}x")

    if normalize is not None:
        normalize = max(1, min(99, normalize))  # Clamp between 1% and 99%
        print(
            f"[*] Normalization ENABLED: All images will be Gamma-corrected to {normalize}% brightness."
        )

    detector = get_face_detector(input_size=target_size)

    files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Filter out files containing the exclusion pattern if provided
    if exclude:
        initial_count = len(files)
        files = [f for f in files if exclude not in f.name]
        excluded_count = initial_count - len(files)
        if excluded_count > 0:
            print(
                f"[*] Excluded {excluded_count} file(s) matching pattern: '{exclude}'"
            )

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
            padding = int(fh * padding_ratio)

            # Use the user-defined pixel padding instead of proportional (fh * 0.6)
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
                cropped_img, target_size, interpolation=cv2.INTER_LANCZOS4
            )

            # --- Target Brightness Normalization ---
            if normalize is not None:
                final_img = normalize_target_brightness(final_img, normalize)

            # Use our new Pillow save function with dynamic size
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
                fallback_img, target_size, interpolation=cv2.INTER_LANCZOS4
            )

            # --- Target Brightness Normalization ---
            if normalize is not None:
                final_img = normalize_target_brightness(final_img, normalize)

            # Use our new Pillow save function with dynamic size
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
        description="Crop student pictures into 1:1 IDs with strict DPI metadata."
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
        "--size",
        type=int,
        default=DEFAULT_TARGET_SIZE[0],
        help=f"Target output size in pixels (creates a square image). Default: {DEFAULT_TARGET_SIZE[0]}",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.6,
        help="Fraction of the face height to use as padding (e.g., 0.6 = 60%%). Default: 0.6",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Skip processing any filenames containing this specific substring pattern.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=None,
        help="Target average brightness percentage (1-99). Uses Gamma correction (Levels adjustment).",
    )
    parser.add_argument(
        "--err-output",
        type=str,
        default="no_face_detected.txt",
        help="Filename to log images where no face was detected (default: no_face_detected.txt).",
    )
    args = parser.parse_args()

    sys.exit(
        main(
            args.target_dir,
            args.output_dir,
            args.err_output,
            args.size,
            args.padding,
            args.exclude,
            args.normalize,
        )
    )
