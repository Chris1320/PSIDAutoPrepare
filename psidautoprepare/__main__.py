import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from psidautoprepare.detector import get_face_detector
from psidautoprepare.info import (
    DEFAULT_TARGET_SIZE,
    IMAGE_EXTENSIONS,
)
from psidautoprepare.postprocessor import normalize_target_brightness


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
    target: str,
    output: str | None,
    err_output: str,
    size: int,
    padding_ratio: float,
    exclude: str | None,
    normalize: int | None,
    soft_stretch: bool,
    black: float,
    white: float,
) -> int:
    input_path = Path(target)
    target_size = (size, size)

    if not input_path.exists():
        print(f"Error: Path '{target}' does not exist.")
        return 1

    single_out_file = None

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            print(f"Error: File '{target}' is not a supported image format.")
            return 1
        files = [input_path]
        base_dir = input_path.parent

        if output:
            out_path = Path(output)
            # If it's an existing directory, or has no file extension, treat as a folder
            if out_path.is_dir() or not out_path.suffix:
                output_dir_path = out_path
                single_out_file = output_dir_path / input_path.name
            # Otherwise, treat it as the exact output filename (e.g. "student.jpg")
            else:
                output_dir_path = out_path.parent
                single_out_file = out_path
        else:
            output_dir_path = base_dir / "output"
            single_out_file = output_dir_path / input_path.name

        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        base_dir = input_path
        output_dir_path = Path(output) if output is not None else base_dir / "output"
        output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory set to: {output_dir_path}")
    print(f"Target Resolution: {size}x{size}px | Padding Ratio: {padding_ratio}x")

    if normalize is not None:
        normalize = max(1, min(99, normalize))  # Clamp between 1% and 99%
        print(
            f"[*] Normalization ENABLED: All images will be Gamma-corrected to {normalize}% brightness."
        )
        if soft_stretch:
            print(f"[*] Soft Stretch ENABLED: Mapping flat images to {black}-{white}.")

    detector = get_face_detector(input_size=target_size)

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
        out_file = (
            single_out_file if single_out_file else (output_dir_path / img_file.name)
        )

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
                final_img = normalize_target_brightness(
                    final_img, normalize, soft_stretch, black, white
                )

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
                final_img = normalize_target_brightness(
                    final_img, normalize, soft_stretch, black, white
                )

            # Use our new Pillow save function with dynamic size
            save_with_dpi(final_img, out_file)

    print("\nProcessing complete!")

    if failed_detections:
        err_file_path = output_dir_path / err_output
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
        "target",
        type=str,
        help="Target directory or specific image file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory (or specific filename if input is a single file) where cropped images will be saved.",
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
        "--soft-stretch",
        action="store_true",
        help="Enable soft contrast stretching for flat images before normalization.",
    )
    parser.add_argument(
        "--black",
        type=float,
        default=25.0,
        help="Target pure black value for soft stretch (0-255). Default: 25.0",
    )
    parser.add_argument(
        "--white",
        type=float,
        default=230.0,
        help="Target pure white value for soft stretch (0-255). Default: 230.0",
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
            args.target,
            args.output,
            args.err_output,
            args.size,
            args.padding,
            args.exclude,
            args.normalize,
            args.soft_stretch,
            args.black,
            args.white,
        )
    )
