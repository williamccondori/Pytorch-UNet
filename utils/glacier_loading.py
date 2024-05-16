import os

import numpy as np
import torch
from torch.utils.data import Dataset


class GlacierDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transformations=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transformations = transformations
        self.images = [x for x in os.listdir(img_folder) if x.endswith(".npy")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.images[idx])
        mask_path = os.path.join(
            self.mask_folder, self.images[idx].replace("IMG", "MASK")
        )

        img = np.load(img_path)
        mask = np.load(mask_path)

        result = {"image": img, "mask": mask}

        if self.transformations:
            result = self.transformations(result)

        return result


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        image = image.transpose((2, 0, 1))

        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


class NormalizeTo0And255(object):
    def __call__(self, sample):
        pass
