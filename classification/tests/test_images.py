from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from classification.background import BackgroundRemover
from classification.errors import ServiceError
from classification.images import decode_image

SETTINGS = SimpleNamespace(max_image_pixels=20_000_000)


def encode(image, format="PNG", **kwargs):
    stream = BytesIO()
    image.save(stream, format=format, **kwargs)
    return stream.getvalue()


@pytest.mark.parametrize(
    "data,media,code",
    [
        (b"", "image/png", "image_empty"),
        (b"broken", "image/png", "image_corrupt"),
        (b"text", "text/plain", "unsupported_image_format"),
        (b"x" * (10 * 1024 * 1024 + 1), "image/png", "image_too_large"),
    ],
    ids=["empty", "corrupt", "unsupported", "oversize"],
)
def test_invalid_upload(data, media, code):
    with pytest.raises(ServiceError) as error:
        decode_image(data, media, SETTINGS)
    assert error.value.code == code


def test_transparent_source_pixels_are_preserved():
    pixels = np.random.default_rng(42).integers(0, 256, (224, 301, 4), dtype=np.uint8)
    image = decode_image(encode(Image.fromarray(pixels)), "image/png", SETTINGS)
    remover = BackgroundRemover.__new__(BackgroundRemover)
    np.testing.assert_array_equal(np.asarray(remover.remove(image)), pixels[:, :, :3])


def test_decode_limits_and_mime():
    for data, media, settings, code in [
        (encode(Image.new("RGB", (224, 224))), "image/jpeg", SETTINGS, "unsupported_image_format"),
        (encode(Image.new("RGB", (223, 224))), "image/png", SETTINGS, "image_too_small"),
        (
            encode(Image.new("RGB", (225, 224))),
            "image/png",
            SimpleNamespace(max_image_pixels=224 * 224),
            "image_too_many_pixels",
        ),
    ]:
        with pytest.raises(ServiceError) as error:
            decode_image(data, media, settings)
        assert error.value.code == code


def test_exif_and_animation():
    exif = Image.Exif()
    exif[274] = 6
    assert decode_image(
        encode(Image.new("RGB", (300, 224)), "JPEG", exif=exif), "image/jpeg", SETTINGS
    ).size == (224, 300)
    frames = [Image.new("RGB", (224, 224), color) for color in ("red", "blue")]
    with pytest.raises(ServiceError) as error:
        decode_image(
            encode(frames[0], save_all=True, append_images=frames[1:]), "image/png", SETTINGS
        )
    assert error.value.code == "animated_image_not_supported"


def test_background_adapter_rejects_wrong_model_and_uniform_image(tmp_path):
    path = tmp_path / "u2net.onnx"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        BackgroundRemover(path)
    with pytest.raises(ServiceError) as error:
        BackgroundRemover.__new__(BackgroundRemover).remove(Image.new("RGB", (224, 224), "white"))
    assert error.value.code == "foreground_not_found"
