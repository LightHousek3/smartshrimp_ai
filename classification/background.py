# SPDX-License-Identifier: MIT
# Adapted from rembg 2.0.67; license retained in THIRD_PARTY_NOTICES.
import numpy as np
import onnxruntime as ort
from PIL import Image

from .constants import U2NET_SHA256
from .errors import ServiceError


class BackgroundRemover:
    def __init__(self, path, threads=2):
        from .model import sha256

        if sha256(path) != U2NET_SHA256:
            raise ValueError("U2Net checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def remove(self, image):
        if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
            return image.convert("RGB")
        image = image.convert("RGB")
        if all(low == high for low, high in image.getextrema()):
            raise ServiceError("foreground_not_found", "No usable foreground in image")
        pixels = np.asarray(image.resize((320, 320), Image.Resampling.LANCZOS))
        values = pixels / max(float(pixels.max()), 1e-6)
        values = (values - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        tensor = values.transpose(2, 0, 1)[None].astype(np.float32)
        try:
            mask = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][:, 0]
            if not np.isfinite(mask).all() or mask.max() <= mask.min():
                raise ValueError("Invalid foreground mask")
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = Image.fromarray((mask.squeeze().clip(0, 1) * 255).astype(np.uint8))
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            return Image.composite(image, Image.new("RGB", image.size, 0), mask)
        except Exception:
            raise ServiceError(
                "background_removal_failed", "Background processing failed", 500
            ) from None
