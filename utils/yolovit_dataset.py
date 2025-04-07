import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

class ROI_Mask_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=384):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # binary

        # Resize both
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))

        image_tensor = TF.to_tensor(image)

        boxes = self.extract_boxes(mask)
        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64)  # assuming one class for now
        }

        return image_tensor, target


    def extract_boxes(self, mask):
        mask_np = np.array(mask)
        boxes = []
        if mask_np.max() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)

        from scipy import ndimage
        labeled, num_objects = ndimage.label(mask_np)

        for obj_id in range(1, num_objects + 1):
            pos = np.where(labeled == obj_id)
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                continue
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            cx = (xmin + xmax) / 2.0 / mask_np.shape[1]
            cy = (ymin + ymax) / 2.0 / mask_np.shape[0]
            w = (xmax - xmin) / mask_np.shape[1]
            h = (ymax - ymin) / mask_np.shape[0]
            boxes.append([cx, cy, w, h])
        return torch.tensor(boxes, dtype=torch.float32)
