import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.detr_vit import DETRViT
from utils.coco_utils import COCODataset

# ==== Config ====
NUM_CLASSES = 2  # UPC_lbl and weight_lbl
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
ANNOTATIONS = "data/annotations/instances_default.json"
IMAGES = "data/images"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    mlflow.start_run()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETRViT(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # No transforms needed here — processor handles it
    dataset = COCODataset(ANNOTATIONS, IMAGES, transforms=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for images, targets in dataloader:
            # Convert tensors back to PIL Images (required for processor)
            from torchvision.transforms.functional import to_pil_image
            images = [to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in images]

            pixel_values, _ = model.prepare_inputs(images)
            pixel_values = pixel_values.to(device)

            # Format labels
            labels = [{"class_labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]

            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mlflow.log_metric("loss", epoch_loss / len(dataloader), step=epoch)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)

    mlflow.end_run()


if __name__ == '__main__':
    train()
