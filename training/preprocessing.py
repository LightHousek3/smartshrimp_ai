import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def prepare_rgb(image: Image.Image, background_removal="none") -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if background_removal == "white_threshold":
        pixels = np.array(image)
        # Explicit opt-in for controlled white backdrops, not semantic segmentation.
        pixels[np.all(pixels >= 240, axis=2)] = 127
        image = Image.fromarray(pixels)
    elif background_removal != "none":
        raise ValueError("Unknown background removal mode")
    return image


def make_transform(size=224, augment=False):
    steps = [transforms.Resize((size, size), antialias=True)]
    if augment:
        steps.extend(
            [transforms.RandomHorizontalFlip(), transforms.RandAugment(num_ops=2, magnitude=9)]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)


def preprocessing_snapshot(config):
    return {
        "input_size": config["input_size"],
        "resize": "bilinear_stretch",
        "color": "RGB",
        "exif_transpose": True,
        "mean": MEAN,
        "std": STD,
        "background_removal": config["background_removal"],
        "aggregation": "mean_softmax",
    }
