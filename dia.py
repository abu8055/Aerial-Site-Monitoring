import torch
import torch.nn as nn
import torchvision
import torchvision.models as models # Library
from torchvision.models.detection import maskrcnn_resnet50_fpn
from ultralytics import YOLO
import cv2
import numpy as np
import time
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_IMG_DIR = #instances_val 
VAL_ANN_FILE = #instances_val.json
MODEL_PATH = #aerial_progress_model.pth 

class AerialProgressModel(nn.Module):
    def __init__(self):
        super(AerialProgressModel, self).__init__()
        swin = models.swin_t(weights='DEFAULT')
        swin.head = nn.Identity()
        self.backbone = swin
        self.feature_dim = 768
        self.adapter = nn.Linear(self.feature_dim, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.regressor = nn.Linear(256, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x)
        features = self.adapter(features)
        features = features.view(b, t, -1)
        gru_out, _ = self.gru(features)
        last_hidden = gru_out[:, -1, :]
        prediction = self.regressor(last_hidden)
        return prediction.squeeze(1)

class FairValDataset(Dataset):
    def __init__(self, root, annFile, limit=5000):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))[:limit]
        
        # Transform A: For Proposed Model (224x224)
        self.transform_model = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform B: For Mask R-CNN (Original Size)
        self.transform_baseline = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target_count = len(self.coco.loadAnns(ann_ids))
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return torch.zeros(5, 3, 224, 224), 0.0, np.zeros((100,100,3), dtype=np.uint8), torch.zeros(3,100,100)

        # 1. Pseudo-Sequence for Proposed Model
        processed_frames = []
        for _ in range(5):
             processed_frames.append(self.transform_model(img))
        seq_tensor = torch.stack(processed_frames)
        
        # 2. Raw for YOLO
        raw_img_np = np.array(img)
        
        # 3. Tensor for Mask R-CNN
        baseline_tensor = self.transform_baseline(img)
        
        return seq_tensor, float(target_count), raw_img_np, baseline_tensor

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    print(f"Running on: {DEVICE}")

    print("Loading AerialProgressModel...")
    my_model = AerialProgressModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        my_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✓ Weights loaded.")
    else:
        print("X Warning: Weights not found! Results will be random.")
    my_model.eval()

    print("\n" + "="*40)
    print("STAGE 1: DIAGNOSTIC SANITY CHECK")
    print("Checking if model is 'lazy' (predicting mean only)...")
    print("="*40)

    # Use small limit for diagnostics
    diag_dataset = FairValDataset(VAL_IMG_DIR, VAL_ANN_FILE, limit=20)
    diag_loader = DataLoader(diag_dataset, batch_size=1, shuffle=True)

    preds_diag = []
    truths_diag = []

    print(f"{'True':<10} | {'Pred':<10} | {'Diff':<10}")
    print("-" * 35)

    for i, (seq, count_label, raw_np, _) in enumerate(diag_loader):
        seq = seq.to(DEVICE)
        with torch.no_grad():
            pred = my_model(seq).item()
        
        true_val = count_label.item()
        preds_diag.append(pred)
        truths_diag.append(true_val)
        
        print(f"{true_val:<10.1f} | {pred:<10.2f} | {abs(true_val-pred):<10.2f}")

        # Save Visual Check
        if i == 0:
            plt.figure(figsize=(6,6))
            plt.imshow(raw_np[0])
            plt.title(f"True: {true_val} | Pred: {pred:.2f}")
            plt.axis('off')
            plt.savefig("visual.png")
            print("\n[Visual] Saved 'visual.png'")

    # Calc Std Dev
    std_dev = np.std(np.array(preds_diag))
    print(f"\nPrediction Standard Deviation: {std_dev:.4f}")

    if std_dev < 0.1:
        print("\n[CRITICAL WARNING] Model is predicting the same number every time!")
        print("Do not proceed to full benchmark. Retrain or debug.")
        exit() 
    else:
        print("\n[PASS] Model predictions vary with input. Proceeding to benchmark.")
