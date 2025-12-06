import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from pycocotools.coco import COCO
from PIL import Image
import argparse
import sys

#Configuration 
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
SEQUENCE_LENGTH = 5  # T=5 frames per pseudo-sequence
IMG_SIZE = 224

#Dataset Class
class MOCSDataset(Dataset):
    """
    MOCS Dataset for Aerial Site Progress Monitoring.
    Generates pseudo-sequences from static images by applying random augmentations.
    Label is the count of objects in the image.
    """
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root (str): Path to the image directory.
            annFile (str): Path to the COCO annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Base transform for resizing and normalization 
        self.base_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Augmentations for pseudo-sequence generation
        # Slight variations to simulate movement/time
        self.augment_transform = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomRotation(degrees=5),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        # Label: Count of objects
        object_count = len(target)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img_path = os.path.join(self.root, path)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}, trying to find it recursively or skipping.")
            raise

        # Generate Pseudo-Sequence
        # T=5 frames. Each frame is an augmented version of the original image.
        
        frames = []
        for _ in range(SEQUENCE_LENGTH):
            # Apply augmentation then base transform
            # Apply random augmentations individually for each frame
            aug_img = self.augment_transform(img)
            processed_frame = self.base_transform(aug_img)
            frames.append(processed_frame)
            
        # Stack frames: (T, C, H, W) -> (5, 3, 224, 224)
        sequence = torch.stack(frames)
        
        return sequence, torch.tensor(float(object_count), dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

# Architecture
class AerialProgressModel(nn.Module):
    def __init__(self):
        super(AerialProgressModel, self).__init__()
        
        # Spatial Core (Backbone): Swin-T
        # We use the default pretrained weights
        swin = models.swin_t(weights='DEFAULT')
        
        swin.head = nn.Identity()
        self.backbone = swin
        
        # Feature size for Swin-T is 768
        self.feature_dim = 768
        
        # 2. Adapter: Linear 768 -> 256
        self.adapter = nn.Linear(self.feature_dim, 256)
        
        # 3. Temporal Core: GRU
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        
        # 4. Head: Regressor
        self.regressor = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (Batch, Sequence, Channels, Height, Width) -> (B, 5, 3, 224, 224)
        b, t, c, h, w = x.shape
        
        # Merge Batch and Sequence dimensions to pass through backbone
        # (B*T, 3, 224, 224)
        x = x.view(b * t, c, h, w)
        
        # Extract features
        # Backbone output: (B*T, 768) (since we replaced head with Identity)
        features = self.backbone(x)
        
        # Adapter
        features = self.adapter(features) # (B*T, 256)
        
        # Reshape back to (B, T, 256) for GRU
        features = features.view(b, t, -1)
        
        # GRU
        # output: (B, T, Hidden), h_n: (NumLayers, B, Hidden)
        gru_out, _ = self.gru(features)
        
        # Take the output of the last time step
        last_hidden = gru_out[:, -1, :] # (B, 256)
        
        # Regressor
        prediction = self.regressor(last_hidden) # (B, 1)
        
        return prediction.squeeze(1) # (B,)

#  Training Loop
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch_idx+1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}")
            
    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch_idx+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for sequences, labels in enumerate(loader):
            # Unpack correctly, enumerate returns index, (data)
            sequences, labels = labels
            
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
    avg_loss = running_loss / len(loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Aerial Site Progress Monitoring Training")
    parser.add_argument('--train-dir', type=str, default='/mnt/c/Users/user/Downloads/instances_train/instances_train', help='Path to training images')
    parser.add_argument('--train-ann', type=str, default='/mnt/c/Users/user/Downloads/instances_train.json', help='Path to training annotations json')
    parser.add_argument('--val-dir', type=str, default='/mnt/c/Users/user/Downloads/instances_val/instances_val', help='Path to validation images')
    parser.add_argument('--val-ann', type=str, default='/mnt/c/Users/user/Downloads/instances_val.json', help='Path to validation annotations json')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch to verify pipeline')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets
    print("Initializing Datasets...")
    train_dataset = MOCSDataset(root=args.train_dir, annFile=args.train_ann)
    val_dataset = MOCSDataset(root=args.val_dir, annFile=args.val_ann)
    
    if args.dry_run:
        print("Dry run mode: reducing dataset size.")
        train_dataset.ids = train_dataset.ids[:10]
        val_dataset.ids = val_dataset.ids[:10]

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    print("Initializing Model...")
    model = AerialProgressModel().to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("Starting Training...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        if args.dry_run:
            print("Dry run completed successfully.")
            break

    print("Training Finished.")
    
    # Save model
    if not args.dry_run:
        save_path = "aerial_progress_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
