import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


MODEL_NAME = "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"
IMG_SIZE = 224
SEED = 42
MIN_DELTA = 0.001


class ImageDS(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label, str(path)


class EarlyStopping:
    """Stop when validation loss does not improve by min_delta for patience epochs."""
    def __init__(self, patience=5, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss):
        if np.isfinite(val_loss) and val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_history(history, report_dir):
    with open(report_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(report_dir / "training_history.png", dpi=160)
    plt.close(fig)


def save_reports(y_true, y_pred, probabilities, paths, classes, report_dir):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - MobileNetV4 Conv Small")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, metric in enumerate(metrics):
        values = [report[class_name][metric] for class_name in classes]
        ax.bar(x + (index - 1) * width, values, width, label=metric)
    ax.set_xticks(x, classes, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics per Class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "class_metrics.png", dpi=160)
    plt.close(fig)

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
    )
    with open(report_dir / "test_metrics.txt", "w", encoding="utf-8") as file:
        file.write("MobileNetV4 Conv Small\n")
        file.write(f"Test images: {len(y_true)}\n")
        file.write(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        file.write(report_text)
        file.write("\nConfusion matrix (rows=actual, columns=predicted):\n")
        file.write(np.array2string(cm))

    with open(report_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with open(report_dir / "predictions_test.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "actual_class", "predicted_class", "confidence_percent"])
        for path, actual, predicted, probs in zip(paths, y_true, y_pred, probabilities):
            writer.writerow([
                path,
                classes[actual],
                classes[predicted],
                f"{max(probs) * 100:.2f}",
            ])

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV4 Conv Small with Early Stopping."
    )
    parser.add_argument("--data-dir", required=True, help="Folder dataset with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate backbone saat fine-tuning")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate classifier saat fine-tuning")
    parser.add_argument("--warmup-epochs", type=int, default=4, help="Epoch awal untuk melatih classifier saja")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder tidak ditemukan: {data_dir}")

    image_folder = datasets.ImageFolder(str(data_dir))
    samples = image_folder.samples
    classes = image_folder.classes
    labels = np.array([label for _, label in samples])

    if len(classes) < 2:
        raise ValueError("Dataset harus memiliki minimal dua folder kelas.")
    class_counts = np.bincount(labels)
    if len(class_counts) != len(classes) or class_counts.min() < 3:
        raise ValueError("Setiap kelas harus memiliki minimal 3 gambar.")

    print("Nama kelas:", classes, flush=True)
    print("Jumlah gambar:", len(samples), flush=True)
    print("Model:", MODEL_NAME, flush=True)
    print("Input:", f"{IMG_SIZE}x{IMG_SIZE}", flush=True)

    indices = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=labels[temp_idx]
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    # Moderate augmentation; validation and test data are not augmented.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = DataLoader(
        ImageDS(train_samples, train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ImageDS(val_samples, eval_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ImageDS(test_samples, eval_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Perangkat:", device, flush=True)
    print(
        f"Data train/validasi/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        flush=True,
    )
    print("Memuat MobileNetV4 pretrained...", flush=True)

    model = timm.create_model(
        MODEL_NAME, pretrained=True, num_classes=len(classes)
    ).to(device)
    print("Model siap; training dimulai.", flush=True)

    # Baseline yang lebih mudah dianalisis: Cross Entropy tanpa label smoothing.
    criterion = nn.CrossEntropyLoss()
    classifier = model.get_classifier()

    # Tahap 1: bekukan backbone, latih classifier saja.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
        threshold=MIN_DELTA, threshold_mode="abs"
    )

    stopper = EarlyStopping(patience=args.patience, min_delta=MIN_DELTA)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = model_dir / "banana_mobilenetv4.pth"
    latest_path = model_dir / "banana_mobilenetv4_latest.pth"
    finetune_started = False

    for epoch in range(args.epochs):
        # Tahap 2: buka seluruh model dan gunakan LR berbeda untuk backbone/head.
        if epoch >= args.warmup_epochs and not finetune_started:
            for parameter in model.parameters():
                parameter.requires_grad = True
            classifier = model.get_classifier()
            head_params = list(classifier.parameters())
            head_ids = {id(parameter) for parameter in head_params}
            backbone_params = [
                parameter for parameter in model.parameters()
                if id(parameter) not in head_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": args.lr},
                    {"params": head_params, "lr": args.head_lr},
                ],
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2,
                threshold=MIN_DELTA, threshold_mode="abs"
            )
            stopper = EarlyStopping(patience=args.patience, min_delta=MIN_DELTA)
            best_val_loss = float("inf")
            finetune_started = True
            print(
                f"\nTahap 2: fine-tuning seluruh model | "
                f"backbone_lr={args.lr:.2e}, classifier_lr={args.head_lr:.2e}",
                flush=True,
            )

        stage = "fine-tuning" if finetune_started else "classifier-only"
        print(
            f"Memulai epoch {epoch + 1}/{args.epochs} ({stage})",
            flush=True,
        )

        model.train()
        if not finetune_started:
            # Jaga backbone dalam eval mode agar statistik BatchNorm pretrained tidak berubah.
            model.eval()
            model.get_classifier().train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError("Train loss NaN/Inf; hentikan dan periksa konfigurasi.")

            loss.backward()
            trainable_params = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            batch_n = targets.size(0)
            train_loss_sum += loss.item() * batch_n
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += batch_n

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss_sum += loss.item() * targets.size(0)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss NaN/Inf.")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        checkpoint = {
            "model_name": MODEL_NAME,
            "image_size": IMG_SIZE,
            "class_names": classes,
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "stage": stage,
        }
        torch.save(checkpoint, latest_path)

        improved = val_loss < best_val_loss - MIN_DELTA
        if improved:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)

        scheduler.step(val_loss)
        save_history(history, report_dir)

        lr_text = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr_text}",
            flush=True,
        )

        # Jangan menghentikan proses sebelum tahap fine-tuning dimulai.
        if finetune_started and stopper.step(val_loss):
            print(
                f"Early stopping pada epoch {epoch + 1}: validation loss "
                f"tidak membaik selama {args.patience} epoch fine-tuning.",
                flush=True,
            )
            break

    if not finetune_started:
        print(
            "Peringatan: tahap fine-tuning tidak berjalan. Naikkan --epochs "
            "agar lebih besar dari --warmup-epochs.",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError("Checkpoint model terbaik tidak terbentuk.")

    # Evaluate the best checkpoint, not the last epoch.
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred, all_probs, test_paths = [], [], [], []
    with torch.inference_mode():
        for images, targets, paths in test_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(probabilities.argmax(axis=1).tolist())
            all_probs.extend(probabilities.tolist())
            test_paths.extend(paths)

    test_accuracy = save_reports(
        y_true, y_pred, all_probs, test_paths, classes, report_dir
    )

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(classes, file, indent=2, ensure_ascii=False)

    print("\n=== TRAINING SELESAI ===", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Best model epoch: {best_checkpoint['epoch']}", flush=True)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)", flush=True)
    print(f"Model terbaik: {best_path.resolve()}", flush=True)
    print(f"Laporan: {report_dir.resolve()}", flush=True)
    for filename in [
        "training_history.png",
        "confusion_matrix.png",
        "class_metrics.png",
        "test_metrics.txt",
        "predictions_test.csv",
        "training_history.json",
        "classification_report.json",
    ]:
        print(" -", report_dir / filename, flush=True)


if __name__ == "__main__":
    main()
