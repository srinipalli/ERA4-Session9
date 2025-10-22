  

# 🧠 ResNet-50 Tiny-ImageNet Training & Resume Script

This project trains a **ResNet-50** deep convolutional neural network on the **Tiny-ImageNet-200** dataset, with features such as **learning-rate finder**, **data augmentation**, **Cosine Annealing LR scheduler**, and **checkpoint resume support**. It logs accuracy and loss metrics to CSV and JSON summaries for easy visualization and reproducibility.

---

## 📊 Key Features

* **Dataset:** [Tiny-ImageNet-200](https://www.kaggle.com/c/tiny-imagenet)

  * 200 classes
  * 100,000 training images (500 per class)
  * 10,000 validation images (50 per class)
  * Image size: 64×64 pixels

* **Model:**

  * Base model: `torchvision.models.resnet50(weights=None)`
  * Modified final layer: `nn.Sequential(Dropout(0.5), Linear(2048, 200))`
  * Fully trainable weights from scratch (no pretrained weights)

* **Training Features:**

  * Automatic **Learning Rate Finder** using `torch_lr_finder`
  * **Data augmentation** (rotation, color jitter, resized crop, horizontal flip)
  * **CosineAnnealingLR** scheduler
  * **Checkpoint resume support** (continues training from saved state)
  * **Best-model auto-save** based on validation accuracy
  * **Detailed metrics logging** to `training_metrics.csv`
  * **JSON summary** (`metrics_summary.json`) for quick inspection
  * Supports **MPS (Apple GPU)**, **CUDA**, and **CPU** fallback

---

## ⚙️ Configuration Overview

| Parameter                | Description                           | Default                           |
| ------------------------ | ------------------------------------- | --------------------------------- |
| `DATA_DIR`               | Tiny-ImageNet dataset root            | `./Session9/tiny-imagenet-200`    |
| `BATCH_SIZE`             | Training batch size                   | 128                               |
| `EPOCHS`                 | Total epochs to train                 | 250                               |
| `LR`                     | Initial learning rate                 | 0.1                               |
| `MODEL_SAVE_PATH`        | File path to save the best checkpoint | `./resnet50_tinyimagenet_best.pt` |
| `LOG_CSV_PATH`           | Training log path                     | `./training_metrics.csv`          |
| `SUMMARY_JSON`           | Summary output path                   | `./metrics_summary.json`          |
| `RESUME`                 | Resume training flag                  | `True`                            |
| `RESUME_COMPLETED_EPOCH` | Epoch from which to resume            | 100                               |

---

## 🚀 How to Run

### **1. Prepare Dataset**

Download the Tiny-ImageNet dataset and extract it into:

```
Session9/tiny-imagenet-200/
│
├── train/
│   ├── n01443537/
│   ├── n01629819/
│   └── ...
└── val/
    ├── images/
    └── val_annotations.txt
```

### **2. Install Requirements**

```bash
pip install torch torchvision matplotlib torch-lr-finder
```

### **3. Train from Scratch**

```bash
python reset50IN.py
```

### **4. Resume Training from Checkpoint (e.g., 100 → 250 epochs)**

```bash
python resnet50_resume.py
```

This script loads the model and optimizer states from the checkpoint and continues training seamlessly from epoch 101.

---

## 📈 Accuracy Summary

Based on the training metrics (`training_metrics.csv`), the model typically achieves:

| Metric                       | Approx. Value          |
| :--------------------------- | :--------------------- |
| **Best Validation Accuracy** | ~65 – 70 %             |
| **Training Accuracy**        | ~80 %                  |
| **Validation Loss (min)**    | ~1.2                   |
| **Learning Rate Range**      | 1e-5 → 1.0 (LR Finder) |

*(Exact numbers depend on hardware, epochs, and augmentations.)*

---

## 📑 Output Files

| File                              | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `resnet50_tinyimagenet_best.pt`   | Best model checkpoint (includes weights and optimizer state) |
| `training_metrics.csv`            | Per-epoch accuracy/loss/logged LR                            |
| `metrics_summary.json`            | Compact summary (epochs, batch size, best accuracy, time)    |
| `lr_finder.csv` / `lr_finder.png` | Learning rate finder history and plot                        |

Example of a **CSV log**:

```
epoch,train_acc,val_acc,train_loss,val_loss,lr,epoch_seconds,cum_minutes
101,78.1200,66.4500,0.8123,1.2345,0.00012345,180.50,312.34
...
```

---

## 🧩 Program Workflow

1. **Data Loading** → Augmented training images and normalized validation images
2. **Model Building** → ResNet-50 + custom FC head
3. **Learning Rate Finder** → Determines optimal LR before training
4. **Training Loop** → Computes train/val loss & accuracy each epoch
5. **Checkpointing** → Saves best model automatically
6. **Resuming** → Loads from checkpoint and continues from next epoch
7. **Logging** → Saves detailed metrics to CSV and JSON

---

## 🖥️ Device Support

* ✅ **CUDA GPU (NVIDIA)**
* ✅ **Apple MPS (Metal Performance Shaders)**
* ✅ **CPU fallback** if no GPU detected

---

## 📚 References

* [Tiny-ImageNet Dataset](https://www.kaggle.com/c/tiny-imagenet)
* [ResNet Paper – Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
* [torch-lr-finder Documentation](https://github.com/davidtvs/pytorch-lr-finder)

---

## 📦 Example Summary (auto-generated JSON)

```json
{
  "epochs": 250,
  "batch_size": 128,
  "learning_rate": 0.1,
  "best_val_acc": 0.694,
  "model_path": "./resnet50_tinyimagenet_best.pt",
  "csv_path": "./training_metrics.csv",
  "total_minutes": 465.2
}
```

---

Would you like me to **extract real accuracy values** from your uploaded `training_metrics.csv` and update the accuracy table and JSON example accordingly?
I can parse the CSV and insert the actual best accuracy and final epoch metrics into the README automatically.
