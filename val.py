import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import json
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
        "image_dir": config["dataset_paths"]["roi_val_im"],
        "mask_dir": config["dataset_paths"]["roi_val_masks"],
        "img_size": img_size
    }

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=2).to(device)
    model.load_state_dict(torch.load(config["model_checkpoint"], map_location=device))
    model.eval()

    dataset = Dataset(**dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    if dataset_type == "ROI_MASK":
        criterion = YOLOLoss(num_classes=2)

    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = [img if isinstance(img, torch.Tensor) else TF.to_tensor(img) for img in images]
            pixel_values = torch.stack(images).to(device)

            if dataset_type == "COCO":
                labels = [{"class_labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            else:
                outputs = model(pixel_values)
                loss = criterion(outputs, targets)

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    evaluate()
