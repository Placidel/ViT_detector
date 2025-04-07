import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=2, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        """
        preds: (B, 7, H, W)
        targets: list of dicts with keys 'boxes' and 'labels'
        """
        B, _, H, W = preds.shape
        device = preds.device

        # Split prediction
        pred = preds.permute(0, 2, 3, 1)  # (B, H, W, 7)
        pred_boxes = pred[..., 0:4]       # (cx, cy, w, h)
        pred_obj = pred[..., 4]          # objectness score
        pred_cls = pred[..., 5:]         # class scores

        # Create target tensors
        target_boxes = torch.zeros_like(pred_boxes, device=device)
        target_obj = torch.zeros_like(pred_obj, device=device)
        target_cls = torch.zeros_like(pred_cls, device=device)

        for b in range(B):
            for box, cls in zip(targets[b]['boxes'], targets[b]['labels']):
                cx, cy, w, h = box
                i = int(cy * H)
                j = int(cx * W)
                target_obj[b, i, j] = 1.0
                target_boxes[b, i, j] = torch.tensor([cx, cy, w, h], device=device)
                target_cls[b, i, j, cls] = 1.0

        # Compute losses
        box_loss = F.l1_loss(pred_boxes[target_obj.bool()], target_boxes[target_obj.bool()], reduction="sum")
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction="sum")
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls[target_obj.bool()], target_cls[target_obj.bool()], reduction="sum")

        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            cls_loss
        )

        return total_loss / B
