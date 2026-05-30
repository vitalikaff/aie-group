"""Датасет Kvasir-SEG и аугментации для задачи сегментации полипов."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def read_split_file(path: str | Path) -> List[str]:
    """Читает train.txt/val.txt и возвращает список имён файлов с расширением .jpg."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() + ".jpg" for line in f if line.strip()]


def build_train_transform(
    image_size: Tuple[int, int],
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.2),
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        ToTensorV2(),
    ])


def build_val_transform(
    image_size: Tuple[int, int],
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> A.Compose:
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=tuple(mean), std=tuple(std)),
        ToTensorV2(),
    ])


class PolypDataset(Dataset):
    """Датасет изображений и бинарных масок Kvasir-SEG."""

    def __init__(
        self,
        file_list: Sequence[str],
        img_dir: str | Path,
        mask_dir: str | Path,
        transforms: A.Compose | None = None,
        return_name: bool = False,
    ) -> None:
        self.file_list = list(file_list)
        self.img_dir = str(img_dir)
        self.mask_dir = str(mask_dir)
        self.transforms = transforms
        self.return_name = return_name

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask)

        mask = mask.unsqueeze(0)
        if self.return_name:
            return image, mask, img_name
        return image, mask
