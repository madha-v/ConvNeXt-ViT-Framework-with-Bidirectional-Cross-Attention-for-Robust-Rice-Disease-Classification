import yaml
import torch
import numpy as np
from data.dataset import build_loaders
from models.hybrid_model import ConvNeXtViTHybridModel
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes = build_loaders(config)
    
    model = ConvNeXtViTHybridModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_model.pth", map_location=device))
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())

    metrics = compute_metrics(labels_all, preds_all)
    print("\n📊 --- FINAL REPOTT OF THE PROJECT ---")
    print(f"Final Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall   : {metrics['recall']:.4f}")
    print(f"Macro F1-Score : {metrics['f1_score']:.4f}")
    
    classes = [str(i) for i in range(num_classes)]
    plot_confusion_matrix(metrics['confusion_matrix'], classes)
    print("📈 Confusion Matrix generated and saved to 'confusion_matrix.png'.")

if __name__ == "__main__":
    main()
