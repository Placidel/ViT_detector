import torch
import torch.nn as nn
import timm


class YOLOViT(nn.Module):
    def __init__(self, num_classes=2, img_size=512):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_384', pretrained=True, features_only=True)
        self.patch_embed_dim = self.backbone.feature_info[-1]['num_chs']

        # Project ViT features to grid-like YOLO prediction map
        self.head = nn.Sequential(
            nn.Conv2d(self.patch_embed_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, (5 + num_classes), 1)  # 5 = [x, y, w, h, obj_conf]
        )

        self.img_size = img_size
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)[-1]  # Use last feature map
        output = self.head(features)
        return output  # shape: (B, 5 + num_classes, H, W)
