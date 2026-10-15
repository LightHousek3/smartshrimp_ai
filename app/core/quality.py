from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.constants import MAX_FILE_BYTES, MIN_IMAGE_SIZE
from app.core.errors import ServiceError

FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


def laplacian_variance(image):
    # Fixed maximum measurement size makes the threshold less resolution dependent.
    gray = image.convert("L")
    gray.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    center = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:] - 4 * center
    )
    return float(laplacian.var())


def decode_checked(data, content_type, settings):
    if not data:
        raise ServiceError("image_empty", "Image is empty")
    if len(data) > MAX_FILE_BYTES:
        raise ServiceError("image_too_large", "Image exceeds 10 MiB")
    if content_type not in FORMATS:
        raise ServiceError("unsupported_image_format", "Only JPEG, PNG and WebP are supported")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format != FORMATS[content_type]:
                raise ServiceError(
                    "unsupported_image_format", "Image format does not match its media type"
                )
            if getattr(probe, "n_frames", 1) != 1:
                raise ServiceError(
                    "animated_image_not_supported", "Animated images are not supported"
                )
            if min(probe.size) < MIN_IMAGE_SIZE:
                raise ServiceError("image_too_small", "Image must be at least 224 x 224 pixels")
            if probe.width * probe.height > settings.max_image_pixels:
                raise ServiceError("image_too_many_pixels", "Decoded image exceeds pixel limit")
            probe.verify()
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            image = ImageOps.exif_transpose(decoded).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServiceError("image_too_many_pixels", "Decoded image exceeds pixel limit") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ServiceError("image_corrupt", "Image cannot be decoded") from None
    if laplacian_variance(image) < settings.blur_threshold:
        raise ServiceError("image_too_blurry", "Image is too blurry; take a sharper photograph")
    return image
