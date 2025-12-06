import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
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
print(f"Benchmark running on: {DEVICE}")

VAL_IMG_DIR = #instances_val 
VAL_ANN_FILE = #instances_val.json
MODEL_PATH = #aerial_progress_model.pth 

class AerialProgressModel(nn.Module):
    def __init__(self):
        super(AerialProgressModel, self).__init__()
        # Load backbone (Swin-T)
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
        # Swin-T requires 224x224 input
        self.transform_model = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform B: For Mask R-CNN (Tensor, but Original Size)
        # DO NOT resize here so the baseline can see small objects clearly
        self.transform_baseline = T.Compose([
            T.ToTensor() # Converts to 0-1 range, keeps original HxW
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target_count = len(self.coco.loadAnns(ann_ids))
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Dummy fallback
            return torch.zeros(5, 3, 224, 224), 0.0, np.zeros((100,100,3), dtype=np.uint8), torch.zeros(3,100,100)

        # 1. Inputs for Proposed Model (Pseudo-Sequence at 224x224)
        processed_frames = []
        for _ in range(5):
             processed_frames.append(self.transform_model(img))
        seq_tensor = torch.stack(processed_frames)
        
        # 2. Inputs for YOLO (Raw Numpy, Original Size)
        # YOLO handles resizing internally, so we give it full quality
        raw_img_np = np.array(img)
        
        # 3. Inputs for Mask R-CNN (Tensor, Original Size)
        baseline_tensor = self.transform_baseline(img)
        
        return seq_tensor, float(target_count), raw_img_np, baseline_tensor

    def __len__(self):
        return len(self.ids)

print("Loading Models...")

# Proposed Model
my_model = AerialProgressModel().to(DEVICE)
if os.path.exists(MODEL_PATH):
    my_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Proposed Model Weights loaded.")
else:
    print("Error: aerial_progress_model.pth not found. Running with random weights.")
my_model.eval()

# YOLOv8
yolo_model = YOLO('yolov8n.pt') 

# Mask R-CNN
mask_rcnn = maskrcnn_resnet50_fpn(weights='DEFAULT').to(DEVICE)
mask_rcnn.eval()

print("\nStarting Benchmark (Accuracy + Speed)")
# limit=5000 ensures we process ALL images in the folder
val_dataset = FairValDataset(VAL_IMG_DIR, VAL_ANN_FILE, limit=5000)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

preds_my = []
preds_yolo = []
preds_mrcnn = []
preds_fd = []
ground_truths = []

# Timer lists
times_my = []
times_yolo = []
times_mrcnn = []

start_total = time.time()

for i, (seq, count_label, raw_np, base_tensor) in enumerate(val_loader):
    seq = seq.to(DEVICE)
    base_tensor = base_tensor.to(DEVICE)
    count_label = count_label.item()
    ground_truths.append(count_label)
    
    # 1. Proposed MODEL (Uses 224x224 Sequence)
    t0 = time.time()
    with torch.no_grad():
        pred = my_model(seq).item()
    times_my.append(time.time() - t0)
    preds_my.append(pred)
    
    # 2. YOLOv8 (Uses Original Resolution Numpy)
    t0 = time.time()
    y_res = yolo_model(raw_np[0].numpy(), verbose=False) 
    times_yolo.append(time.time() - t0)
    y_count = len(y_res[0].boxes)
    preds_yolo.append(float(y_count))

    # 3. Mask R-CNN (Uses Original Resolution Tensor)
    t0 = time.time()
    with torch.no_grad():
        out = mask_rcnn(base_tensor) # Input: [C, H, W]
        m_count = len([s for s in out[0]['scores'] if s > 0.5])
    times_mrcnn.append(time.time() - t0)
    preds_mrcnn.append(float(m_count))

    # 4. Frame Differencing (Baseline)
    # Using 224 version for speed/simplicity as pixel diff is scale-agnostic roughly
    f1 = (seq[0, 0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    f2 = (seq[0, -1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    pixel_change = np.sum(thresh) / (224*224*255)
    preds_fd.append(5.0 if pixel_change > 0.1 else 0.0)

    if i % 100 == 0:
        elapsed = time.time() - start_total
        print(f"Processed {i}/{len(val_loader)} images... ({elapsed:.1f}s)")

mae_my = mean_absolute_error(ground_truths, preds_my)
mae_yolo = mean_absolute_error(ground_truths, preds_yolo)
mae_mrcnn = mean_absolute_error(ground_truths, preds_mrcnn)
mae_fd = mean_absolute_error(ground_truths, preds_fd)

# Calculate FPS (1 / Average Time)
fps_my = 1.0 / np.mean(times_my)
fps_yolo = 1.0 / np.mean(times_yolo)
fps_mrcnn = 1.0 / np.mean(times_mrcnn)
fps_fd = 1000.0 # Dummy high value for frame diff (instant)

print("\n--- RESULTS ---")
print(f"Proposed Model: MAE={mae_my:.4f} | FPS={fps_my:.1f}")
print(f"YOLOv8:     MAE={mae_yolo:.4f} | FPS={fps_yolo:.1f}")
print(f"Mask R-CNN: MAE={mae_mrcnn:.4f} | FPS={fps_mrcnn:.1f}")
print(f"Frame Diff: MAE={mae_fd:.4f}")


models = ["Frame Diff", "Mask R-CNN", "YOLOv8", "Proposed Model"]
maes = [mae_fd, mae_mrcnn, mae_yolo, mae_my]
fps_scores = [0, fps_mrcnn, fps_yolo, fps_my] # 0 for Frame Diff to clean up graph

fig, ax1 = plt.subplots(figsize=(12, 7))
x = np.arange(len(models))
width = 0.35

# Bar 1: Accuracy (Error) - Left Axis
color1 = 'tab:blue'
ax1.set_ylabel('Mean Absolute Error (Lower is Better)', color=color1, fontsize=12)
bars1 = ax1.bar(x - width/2, maes, width, color=color1, label='Error (MAE)')
ax1.tick_params(axis='y', labelcolor=color1)

# Bar 2: Speed (FPS) - Right Axis
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Inference Speed (FPS) (Higher is Better)', color=color2, fontsize=12)
bars2 = ax2.bar(x + width/2, fps_scores, width, color=color2, label='Speed (FPS)')
ax2.tick_params(axis='y', labelcolor=color2)

# Labels and Legend
plt.title('Benchmark: Accuracy vs. Speed', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

# Add text labels on bars for readability
for i, v in enumerate(maes):
    ax1.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center', color='black', fontweight='bold')
for i, v in enumerate(fps_scores):
    if v > 0:
        ax2.text(i + width/2, v + 1, f"{v:.1f}", ha='center', color='black', fontweight='bold')

output_file = "benchmark_dual.png"
plt.savefig(output_file)
print(f"Graph saved to {output_file}")