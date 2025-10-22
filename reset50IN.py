import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time, os, csv, json

# NEW: for saving the LR plot in headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NEW: official LR Finder package
from torch_lr_finder import LRFinder
# =============================
# CONFIGURATION (edit here)
# =============================
DATA_DIR   = "./Session9/tiny-imagenet-200"  # path to TinyImageNet
BATCH_SIZE      = 128
EPOCHS          = 250
LR              = 0.1
NUM_WORKERS     = 4
MODEL_SAVE_PATH = "./resnet50_tinyimagenet_best.pt"
LOG_CSV_PATH    = "./training_metrics.csv"
SUMMARY_JSON    = "./metrics_summary.json"
# =============================

# LR Finder settings
RUN_LR_FINDER   = True
LRF_START       = 1e-5
LRF_END         = 1.0
LRF_ITERS       = 800              # reduce if your loader is short
LRF_STEP_MODE   = "exp"            # "exp" or "linear"
LRF_CSV_PATH    = "./lr_finder.csv"
LRF_PNG_PATH    = "./lr_finder.png"

MODEL_SAVE_PATH = "./resnet50_tinyimagenet_best.pt"
LOG_CSV_PATH    = "./training_metrics.csv"
SUMMARY_JSON    = "./metrics_summary.json"
SEED            = 42
# =============================

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_loaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  

    train_tfms = transforms.Compose([
        transforms.RandomRotation(15), # New
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # New
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"),   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    return train_loader, val_loader

def build_model_old(num_classes=200):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
def build_model(num_classes=200):
    model = models.resnet50(weights=None)
    # 1. Create a sequential block for the new head
    new_fc = nn.Sequential(
        nn.Dropout(p=0.5), # Add a dropout layer (0.5 is a common starting point)
        nn.Linear(model.fc.in_features, num_classes)
    )
    # 2. Replace the existing fc layer
    model.fc = new_fc
    return model
def ensure_csv(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def append_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def run_lr_finder(model, optimizer, train_loader, device):
    """
    Uses torch-lr-finder to sweep LR and pick a starting LR.
    Saves lr_finder.csv + lr_finder.png and returns suggested_lr.
    Heuristic: lr_at_min_loss / 10.
    """
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    # range_test modifies model/opt but reset() restores them
    lr_finder.range_test(
        train_loader,
        end_lr=LRF_END,
        start_lr=LRF_START,
        num_iter=LRF_ITERS,
        step_mode=LRF_STEP_MODE,
        diverge_th=5.0
    )

    # Save plot
    try:
        ax = lr_finder.plot(log_lr=True)  # returns a matplotlib Axes
        fig = ax.get_figure()
        fig.savefig(LRF_PNG_PATH, bbox_inches="tight")
        plt.close(fig)
        print(f"[LRFinder] Plot saved: {LRF_PNG_PATH}")
    except Exception as e:
        print(f"[LRFinder] Plot save skipped: {e}")

    # Save CSV
    hist = lr_finder.history
    ensure_csv(LRF_CSV_PATH, ["iter", "lr", "loss"])
    for i, (lr, loss) in enumerate(zip(hist["lr"], hist["loss"]), start=1):
        append_row(LRF_CSV_PATH, [i, f"{lr:.8f}", f"{loss:.6f}"])
    print(f"[LRFinder] CSV saved: {LRF_CSV_PATH}")

    # Suggest LR = lr_at_min_loss / 10
    losses = np.array(hist["loss"])
    lrs    = np.array(hist["lr"])
    min_idx = int(np.argmin(losses))
    suggested_lr = float(lrs[min_idx] / 10.0)
    print(f"[LRFinder] min loss at iter {min_idx+1}, lr={lrs[min_idx]:.6f} -> suggested={suggested_lr:.6f}")

    # Reset weights/optimizer to pre-test state
    lr_finder.reset()
    return suggested_lr


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total  # (loss, acc)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total  # (loss, acc)

def ensure_csv(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    if write_header:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

def append_row(path, row):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)

    # Prepare loaders, model, optim, sched
    train_loader, val_loader = build_loaders()
    model = build_model().to(device)
        # --- Sanity checks (run once) ---
    print("Train classes:", len(train_loader.dataset.classes))
    print("Val classes:  ", len(val_loader.dataset.classes))
    print(" Start Training ")
# Class name / index alignment check
    train_idx = train_loader.dataset.class_to_idx
    val_idx   = val_loader.dataset.class_to_idx
    if train_idx != val_idx:
        print("\n[WARNING] class_to_idx mismatch between train and val!")
        # Show 10 sample mappings from each so you can see the difference
        print("train.class_to_idx (first 10):", list(train_idx.items())[:10])
        print("val.class_to_idx   (first 10):", list(val_idx.items())[:10])


    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)
   
   # ---- Run LR Finder (optional) ----
    if RUN_LR_FINDER:
        print("\n[LRFinder] Running LR sweep ...")
        suggested_lr = run_lr_finder(model, optimizer, train_loader, device)
        # Apply the found LR
        for pg in optimizer.param_groups:
            pg["lr"] = suggested_lr
        print(f"[LRFinder] Applied startup LR: {suggested_lr:.6f}\n")

    # CSV logging setup
    header = ["epoch", "train_acc", "val_acc", "train_loss", "val_loss", "lr", "epoch_seconds", "cum_minutes"]
    ensure_csv(LOG_CSV_PATH, header)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    t0 = time.time()
    cum_minutes = 0.0

    for epoch in range(1, EPOCHS + 1):
        e_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        e_end = time.time()

        epoch_seconds = e_end - e_start
        cum_minutes = (time.time() - t0) / 60.0
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else LR

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.5f}")

        if epoch in [1, 5]:
            print(f"\n>>> Highlight Epoch {epoch}: Train Acc={train_acc*100:.2f}%, "
                f"Val Acc={val_acc*100:.2f}% <<<\n")

        # Append to CSV
        append_row(LOG_CSV_PATH, [
            epoch,
            f"{train_acc*100:.6f}",
            f"{val_acc*100:.6f}",
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{current_lr:.8f}",
            f"{epoch_seconds:.2f}",
            f"{cum_minutes:.2f}",
        ])

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr
            }, MODEL_SAVE_PATH)

    total_minutes = (time.time() - t0) / 60.0
    print(f"\nTraining Completed in {total_minutes:.2f} minutes")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Metrics CSV: {LOG_CSV_PATH}")

    # Write a small JSON summary (optional but handy)
    summary = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "best_val_acc": best_acc,
        "model_path": MODEL_SAVE_PATH,
        "csv_path": LOG_CSV_PATH,
        "total_minutes": round(total_minutes, 2)
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
