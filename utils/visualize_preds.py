import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import json
from utils.yolovit_dataset import ROI_Mask_Dataset
from models.yolo_vit import YOLOViT
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

def visualize_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open("config.json") as f:
        config = json.load(f)

    model = YOLOViT(num_classes=2).to(device)
    model.load_state_dict(torch.load(config["model_checkpoint"], map_location=device))
    model.eval()

    dataset = ROI_Mask_Dataset(
        image_dir=config["dataset_paths"]["roi_val_im"],
        mask_dir=config["dataset_paths"]["roi_val_masks"],
        img_size=config["img_size"]
    )

    class_names = ["UPC_lbl", "weight_lbl"]
    to_pil = ToPILImage()

    for i in range(5):  # Show 5 predictions
        image, _ = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)[0]  # shape: (7, H, W)

        pred = pred.permute(1, 2, 0)  # (H, W, 7)
        pred = pred.reshape(-1, 7)    # flatten all boxes

        boxes = pred[:, :4]
        confs = torch.sigmoid(pred[:, 4])
        classes = torch.softmax(pred[:, 5:], dim=1).argmax(dim=1)

        # Filter low confidence
        keep = confs > 0.3
        boxes = boxes[keep]
        classes = classes[keep]
        confs = confs[keep]

        if boxes.shape[0] == 0:
            print("No objects found.")
            continue

        # Convert cx,cy,w,h -> x1,y1,x2,y2
        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        labels = [f"{class_names[int(c)]}: {conf.item():.2f}" for c, conf in zip(classes, confs)]

        img = (image * 255).byte()
        drawn = draw_bounding_boxes(img, boxes_xyxy, labels, width=2)
        to_pil(drawn).show()

if __name__ == '__main__':
    visualize_predictions()
