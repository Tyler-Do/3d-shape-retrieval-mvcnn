# datasets.py
import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CachedMultiViewShapeDataset(Dataset):
    """
    Dataset for cached multi–view PNG images.

    Each sample corresponds to one 3D object. We store:
        - obj_dir: path to folder containing view_XX.png files
        - label:   integer class id

    Folder structure is assumed to be:
        cache_root/
            train/
                chair/
                    chair_0001/
                        view_00.png, ..., view_07.png
                desk/
                    ...
            test/
                ...

    This is exactly the structure produced by `render_cache.py`.
    """

    def __init__(self, cache_root: str, split: str = "train", image_size: int = 128):
        super().__init__()
        self.root = os.path.join(cache_root, split)

        # Class folders: chair, desk, dresser, ...
        self.classes: List[str] = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all object directories (one dir = one 3D object).
        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for obj_name in os.listdir(cls_dir):
                obj_dir = os.path.join(cls_dir, obj_name)
                if os.path.isdir(obj_dir):
                    label = self.class_to_idx[cls]
                    self.samples.append((obj_dir, label))

        # We normalize to [-1, 1] to match the training code.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        obj_dir, label = self.samples[idx]

        # Collect all views for this object and sort them to keep a stable order.
        view_files = sorted(
            f for f in os.listdir(obj_dir) if f.endswith(".png")
        )

        imgs = []
        for vf in view_files:
            img_path = os.path.join(obj_dir, vf)
            img = Image.open(img_path).convert("RGB")
            imgs.append(self.transform(img))

        # We stack along a new dimension: [V, C, H, W]
        views_tensor = torch.stack(imgs, dim=0)
        return views_tensor, label


def make_loaders(cache_root: str,
                 img_size: int,
                 batch_size: int,
                 num_workers: int,
                 val_split: float = 0.2):
    """
    Helper that builds train / val / test DataLoaders from the cached views.

    We split the 'train' split into train+val subsets, and keep 'test'
    as the official test set.
    """
    full_train_ds = CachedMultiViewShapeDataset(cache_root, split="train",
                                                image_size=img_size)

    # Compute split sizes.
    val_len = int(len(full_train_ds) * val_split)
    train_len = len(full_train_ds) - val_len
    train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

    test_ds = CachedMultiViewShapeDataset(cache_root, split="test",
                                          image_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
