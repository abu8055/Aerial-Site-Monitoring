import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.data import Dataset
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
        self.adapter = nn.Linear(768, 256)
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
        return self.regressor(last_hidden).squeeze(1)

class FairValDataset(Dataset):
    def __init__(self, root, annFile):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform_model = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            return None 

        # Pseudo-Sequence
        processed_frames = [self.transform_model(img) for _ in range(5)]
        seq_tensor = torch.stack(processed_frames)
        
        # Raw image for plotting
        raw_img_np = np.array(img)
        
        return seq_tensor, float(target_count), raw_img_np

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    print(f"Generating 5 separate visuals on {DEVICE}...")
    
    # Load Model
    model = AerialProgressModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    else:
        print("Error: Model weights not found!")
        exit()
    model.eval()

    # Load Data
    dataset = FairValDataset(VAL_IMG_DIR, VAL_ANN_FILE)
    
    # Pick 5 random indices
    indices = random.sample(range(len(dataset)), 5)
    
    for i, idx in enumerate(indices):
        data = dataset[idx]
        if data is None: continue
        
        seq_tensor, true_count, raw_img = data
        seq_tensor = seq_tensor.unsqueeze(0).to(DEVICE) 
        
        with torch.no_grad():
            pred_count = model(seq_tensor).item()
            
        # Create individual plot
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_img)
        plt.axis('off')
        
        # Color code title
        error = abs(true_count - pred_count)
        color = 'green' if error < 1.0 else 'red'
        
        plt.title(f"Example {i+1}\nTrue: {int(true_count)} | Pred: {pred_count:.2f}", 
                  color=color, fontsize=16, fontweight='bold', backgroundcolor='white')

        # Save individual file
        filename = f"check_{i+1}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 
        
        print(f"Saved {filename} (Error: {error:.2f})")

    print("\nAll 5 images generated successfully.")