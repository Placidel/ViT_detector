import torch
import torch.nn as nn
from transformers import DetrImageProcessor, DetrForObjectDetection

class DETRViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Set correct number of labels inside config FIRST
        self.model.config.num_labels = num_classes

        # Update classification head
        hidden_size = self.model.class_labels_classifier.in_features
        self.model.class_labels_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    def prepare_inputs(self, images):
        return self.processor(images=images, return_tensors="pt", padding=True, do_rescale=False)
