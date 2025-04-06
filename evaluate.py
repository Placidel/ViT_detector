import os
import json
import torch
from torchvision import transforms
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

from models.detr_vit import DETRViT
from utils.coco_utils import COCODataset

ANNOTATIONS = "data/annotations/instances_default.json"
IMAGES = "data/images"
CHECKPOINT = "checkpoints/model_epoch_19.pth"  # Change to your desired checkpoint

def prepare_coco_predictions(model, dataset):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            image, target = dataset[idx]
            img_id = list(dataset.images.keys())[idx]
            pixel_values, _ = model.prepare_inputs([image])
            pixel_values = pixel_values.cuda() if torch.cuda.is_available() else pixel_values

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]  # Normalized format

            probs = logits.softmax(-1)
            scores, labels = probs.max(-1)

            for box, score, label in zip(boxes, scores, labels):
                if label.item() >= len(dataset.categories):  # Skip 'no object'
                    continue
                coco_box = box.cpu().numpy()
                xmin, ymin, xmax, ymax = coco_box
                width = xmax - xmin
                height = ymax - ymin
                all_preds.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(xmin), float(ymin), float(width), float(height)],
                    "score": float(score)
                })

    return all_preds

def run_coco_eval(predictions, annotation_file):
    with open("coco_preds.json", "w") as f:
        json.dump(predictions, f)

    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes("coco_preds.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = COCODataset(ANNOTATIONS, IMAGES, transforms=transform)

    model = DETRViT(num_classes=3)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))
    model = model.cuda() if torch.cuda.is_available() else model

    predictions = prepare_coco_predictions(model, dataset)
    run_coco_eval(predictions, ANNOTATIONS)
