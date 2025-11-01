import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 10


# Dataset class
class CactusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transforms with augmentation
transform_train = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

# Load data
train_csv = "./input/train.csv"
train_dir = "./input/train/"
test_dir = "./input/test/"

# Split data into train and validation sets
train_df = pd.read_csv(train_csv)
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Save temporary CSV files
train_data.to_csv("./working/train_temp.csv", index=False)
val_data.to_csv("./working/val_temp.csv", index=False)

# Create datasets
train_dataset = CactusDataset(
    csv_file="./working/train_temp.csv", root_dir=train_dir, transform=transform_train
)
val_dataset = CactusDataset(
    csv_file="./working/val_temp.csv", root_dir=train_dir, transform=transform_val
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=2, verbose=True
)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs.squeeze())
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    auc_score = roc_auc_score(val_labels, val_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation AUC: {auc_score:.4f}")

    # Step the scheduler
    scheduler.step(auc_score)

# Generate predictions for the test set
test_dataset = CactusDataset(
    csv_file="./input/sample_submission.csv", root_dir=test_dir, transform=transform_val
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

submission = pd.read_csv("./input/sample_submission.csv")
model.eval()
test_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs.squeeze())
        test_preds.extend(preds.cpu().numpy())

# Save submission
submission["has_cactus"] = test_preds
submission.to_csv("./submission/submission.csv", index=False)
