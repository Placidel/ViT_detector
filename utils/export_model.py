import torch
import json
from models.yolo_vit import YOLOViT

def export_model():
    with open("config.json") as f:
        config = json.load(f)

    model = YOLOViT(num_classes=2)
    model.load_state_dict(torch.load(config["model_checkpoint"], map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, config["img_size"], config["img_size"])

    # === TorchScript ===
    traced = torch.jit.trace(model, dummy_input)
    traced.save("exports/yolovit_traced.pt")
    print("✅ TorchScript saved to exports/yolovit_traced.pt")

    # === ONNX ===
    torch.onnx.export(
        model, dummy_input, "exports/yolovit.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=17
    )
    print("✅ ONNX saved to exports/yolovit.onnx")

if __name__ == '__main__':
    export_model()
