import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import GradScaler, autocast

from data.dataset import build_loaders
from models.hybrid_model import ConvNeXtViTHybridModel
from utils.ema import EMAModel
from utils.metrics import compute_metrics


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_layerwise_optimizer(model, base_lr, weight_decay):
    parameter_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": base_lr * 0.1
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": base_lr
        }
    ]

    return optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay
    )


def main():

    # =========================
    # Load Configuration
    # =========================

    config = load_config()

    os.makedirs(
        config["training"]["checkpoint_dir"],
        exist_ok=True
    )


    # =========================
    # Device Configuration
    # =========================

    device = torch.device(
        config["training"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    print("=" * 50)
    print(f"Device       : {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"AMP Enabled  : {use_amp}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    print("=" * 50)


    # =========================
    # Dataset
    # =========================

    print("📂 Building DataLoaders...")

    train_loader, val_loader, num_classes = build_loaders(config)

    print(f"Number of classes: {num_classes}")
    print(f"Training batches : {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")


    # =========================
    # Model
    # =========================

    print("🧠 Loading Hybrid ConvNeXt + ViT Model...")

    model = ConvNeXtViTHybridModel(
        num_classes=num_classes,
        dropout=config["model"]["dropout"]
    ).to(device)


    # =========================
    # Loss
    # =========================

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["model"]["label_smoothing"]
    )


    # =========================
    # Optimizer
    # =========================

    optimizer = get_layerwise_optimizer(
        model,
        config["training"]["base_lr"],
        config["training"]["weight_decay"]
    )


    # =========================
    # Scheduler
    # =========================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6
    )


    # =========================
    # AMP
    # =========================

    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )


    # =========================
    # EMA
    # =========================

    ema = EMAModel(
        model,
        decay=config["training"]["ema_decay"]
    )


    best_f1 = 0.0


    # =========================
    # Training Loop
    # =========================

    for epoch in range(config["training"]["epochs"]):

        print(
            f"\n🚀 Epoch [{epoch + 1}/{config['training']['epochs']}]"
        )

        model.train()

        running_loss = 0.0


        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=use_amp
            )

            labels = labels.to(
                device,
                non_blocking=use_amp
            )

            optimizer.zero_grad(set_to_none=True)


            # =========================
            # Forward Pass
            # =========================

            with autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            # =========================
            # Backward Pass
            # =========================

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            ema.update()

            running_loss += loss.item()


            # =========================
            # Training Progress
            # =========================

            if (batch_idx + 1) % 10 == 0:

                print(
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )


        # =========================
        # Scheduler
        # =========================

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"📉 Average Training Loss: {epoch_loss:.4f}"
        )


        # =========================
        # Validation
        # =========================

        ema.apply_shadow()

        model.eval()

        preds_all = []
        labels_all = []


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=use_amp
                )

                with autocast(
                    device_type=device.type,
                    enabled=use_amp
                ):

                    outputs = model(images)


                predictions = outputs.argmax(dim=1)

                preds_all.extend(
                    predictions.cpu().numpy()
                )

                labels_all.extend(
                    labels.numpy()
                )


        # =========================
        # Metrics
        # =========================

        metrics = compute_metrics(
            labels_all,
            preds_all
        )


        print(
            f"📊 Epoch [{epoch + 1:02d}] "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1-Score: {metrics['f1_score']:.4f}"
        )


        # =========================
        # Save Best Model
        # =========================

        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]

            checkpoint_path = os.path.join(
                config["training"]["checkpoint_dir"],
                "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"✨ Best model saved! F1: {best_f1:.4f}"
            )


        # =========================
        # Restore EMA
        # =========================

        ema.restore()


    print("\n🎉 Training Completed Successfully!")

    print(
        f"🏆 Best Macro F1-Score: {best_f1:.4f}"
    )


if __name__ == "__main__":
    main()