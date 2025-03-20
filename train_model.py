import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from config import set_dataset_paths, BATCH_SIZE, LEARNING_RATE, MOMENTUM, EPOCHS
from prepare_dataset import ImageTensorDataset
from models.model_ann import ANNModel
import pandas as pd
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
train_dataset_path, _ = set_dataset_paths()
train_dataset = ImageTensorDataset(train_dataset_path)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

input_size = 224 * 224 * 3
num_classes = len(train_dataset.class_to_idx)

model = ANNModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Pastikan folder results ada
if not os.path.exists('results'):
    os.makedirs('results')

# Training
def train():
    model.train()
    loss_data = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
        # Simpan data loss
        loss_data.append({"Epoch": epoch+1, "Loss": avg_loss})

    # Menyimpan model
    torch.save(model.state_dict(), "models/model_ann.pth")
    print("Model berhasil disimpan di models/model_ann.pth")

    # Menyimpan data loss ke CSV
    df = pd.DataFrame(loss_data)
    df.to_csv("results/loss_data.csv", index=False)
    print("Data loss berhasil disimpan di results/loss_data.csv")

if __name__ == "__main__":
    train()