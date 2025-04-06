# DETR Vision Transformer for Object Detection

This project implements a Vision Transformer (ViT)-based DETR model using the HuggingFace Transformers library. It's trained on a COCO-format dataset with two object classes: `UPC_lbl` and `weight_lbl`.

## 🧰 Requirements
- Python 3.9+
- CUDA 12.6 compatible GPU (e.g., RTX 4070)
- PyTorch 2.2+

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Dataset Format
Place the data as follows:
```
data/
├── images/                      # All image files
└── annotations/
    └── instances_default.json   # COCO annotation file
```

## 🚀 Training
```bash
python train.py
```
Model checkpoints are saved in the `checkpoints/` folder and tracked with MLflow.

## 📈 Logging
MLflow tracks:
- Loss per epoch
- Model checkpoints as artifacts
- Hyperparameters

## 📄 Evaluation
Run COCO mAP evaluation:
```bash
python evaluate.py
```

## 🔧 Configuration
- `train.py` uses default hyperparameters, can be modified inline.
- Adjust `NUM_CLASSES`, `LEARNING_RATE`, `EPOCHS`, and checkpoint paths.

---

This scaffold uses HuggingFace’s `DetrForObjectDetection` with an updated classification head for your two categories plus background. Expand the loss function and evaluation as needed for production.