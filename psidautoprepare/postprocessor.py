import math

import cv2
import numpy as np


def normalize_target_brightness(
    cv2_image: cv2.typing.MatLike,
    target_percentage: int,
    apply_soft_stretch: bool = False,
    black_point: float = 25.0,
    white_point: float = 230.0,
) -> cv2.typing.MatLike:
    """
    Uses Gamma Correction to non-linearly adjust the midtones of the image
    so the overall average brightness matches the target percentage.
    If enabled, applies a soft contrast stretch to standardize the dynamic range.
    """

    if apply_soft_stretch:
        # 1. Detect dynamic range
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        p1, p99 = np.percentile(gray, (1, 99))
        dynamic_range = p99 - p1

        # Soft stretch: map to user-defined points instead of 0-255 to keep it looking natural
        # Safety check added to prevent division by zero on solid color images
        if dynamic_range > 0:
            alpha = (white_point - black_point) / dynamic_range
            beta = black_point - (p1 * alpha)
            cv2_image = cv2.convertScaleAbs(cv2_image, alpha=alpha, beta=beta)

    # Recalculate grayscale mean for the Gamma step
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    current_mean = cv2.mean(gray)[0]

    # Safety check: if image is entirely pitch black or blindingly white, skip math
    if current_mean <= 1 or current_mean >= 254:
        return cv2_image

    # 2. Calculate target mean (0 to 255)
    target_mean = (target_percentage / 100.0) * 255.0

    # 3. Calculate the specific Gamma curve needed to bend the current mean to the target mean
    # Formula: target = current ^ gamma  =>  gamma = log(target) / log(current) (scaled 0-1)
    gamma = math.log(target_mean / 255.0) / math.log(current_mean / 255.0)

    # 4. Build a Look-Up Table (LUT) for blazing-fast mapping
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    final_bgr = cv2.LUT(cv2_image, lut)

    # 5. Denoise (Fixes grain if we stretched deep shadows heavily)
    if gamma < 0.8:
        final_bgr = cv2.bilateralFilter(final_bgr, d=5, sigmaColor=25, sigmaSpace=25)

    return final_bgr
