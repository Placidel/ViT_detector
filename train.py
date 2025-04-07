import os
import json
import mlflow
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from tqdm import tqdm

# === Load config ===
with open("config.json") as f:
    config = json.load(f)

dataset_type = config["dataset"]
img_size = config.get("img_size", 384)

if dataset_type == "COCO":
    from models.detr_vit import DETRViT as Model
    from utils.coco_utils import COCODataset as Dataset
    dataset_kwargs = {
        "annotation_file": "data/annotations/instances_default.json",
        "image_dir": "data/images",
        "transforms": None
    }
elif dataset_type == "ROI_MASK":
    from models.yolo_vit import YOLOViT as Model
    from utils.yolovit_dataset import ROI_Mask_Dataset as Dataset
    from utils.yolo_loss import YOLOLoss
    dataset_kwargs = {
        "image_dir": config["dataset_paths"]["roi_train_im"],
        "mask_dir": config["dataset_paths"]["roi_train_masks"],
        "img_size": img_size
    }

# === Training config ===
NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 2e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === TensorBoard setup ===
writer = SummaryWriter(log_dir="runs/yolovit_experiment")

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    mlflow.start_run()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Model(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dataset = Dataset(**dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    if dataset_type == "ROI_MASK":
        criterion = YOLOLoss(num_classes=NUM_CLASSES)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")

        for images, targets in pbar:
            images = [img if isinstance(img, torch.Tensor) else TF.to_tensor(img) for img in images]
            pixel_values = torch.stack(images).to(device)
            labels = [{"class_labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]

            optimizer.zero_grad()
            if dataset_type == "COCO":
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            elif dataset_type == "ROI_MASK":
                outputs = model(pixel_values)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)

    writer.close()
    mlflow.end_run()

if __name__ == '__main__':
    train()
