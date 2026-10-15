from PIL import Image
from torch.utils.data import Dataset

from app.core.constants import CLASSES
from training.preprocessing import make_transform, prepare_rgb


class ShrimpDataset(Dataset):
    def __init__(self, manifest, root, split, config, augment=False):
        self.samples = [s for s in manifest["samples"] if s["split"] == split]
        self.root = root
        self.background = config.background_removal
        self.transform = make_transform(config.input_size, augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        with Image.open(self.root / row["path"]) as image:
            if min(image.size) < 224:
                raise ValueError(f"Dataset image below 224 pixels: {row['id']}")
            tensor = self.transform(prepare_rgb(image, self.background))
        return tensor, CLASSES.index(row["label"])
