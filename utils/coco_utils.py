import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        with open(annotation_file) as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        self.transforms = transforms

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

        # Keep only image_ids that have at least one annotation
        self.image_ids = [img_id for img_id in self.images if img_id in self.img_to_anns]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.img_to_anns.get(img_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
