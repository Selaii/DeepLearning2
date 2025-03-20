import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from config import set_dataset_paths
from models.model_ann import ANNModel
from prepare_dataset import ImageTensorDataset

# Load Dataset Path
_, test_dataset_path = set_dataset_paths()

# Load Dataset untuk mendapatkan class_to_idx
test_dataset = ImageTensorDataset(test_dataset_path)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# Load Data
df = pd.read_csv('results/test_results.csv')

# Load Model
input_size = 224 * 224 * 3
num_classes = len(df['Actual Label'].unique())
model = ANNModel(input_size, num_classes)
model.load_state_dict(torch.load("models/model_ann.pth"))
model.eval()

# Transform untuk gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Buat figure dan subplots
num_images = 26  # Jumlah gambar yang ingin ditampilkan
rows = 5  # Jumlah baris untuk gambar
cols = 6  # Jumlah kolom untuk gambar
fig = plt.figure(figsize=(18, 20))  # Ukuran figure disesuaikan

# Plot Training Loss di bagian atas
ax1 = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)  # Baris pertama untuk grafik loss
loss_data = np.loadtxt('results/loss_data.csv', delimiter=',', skiprows=1)
epochs, losses = loss_data[:, 0], loss_data[:, 1]
ax1.plot(epochs, losses, marker='o', linestyle='-')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Training Error over Epochs")
ax1.grid()

# Plot Gambar Hasil Prediksi di bagian bawah
for i in range(num_images):
    try:
        # Konversi Label ke Nama Kelas
        actual_label = int(df['Actual Label'][i])
        predicted_label = int(df['Predicted Label'][i])
        image_path = df['Image Path'][i]

        actual_label_name = idx_to_class[actual_label]
        predicted_label_name = idx_to_class[predicted_label]

        # Load Gambar
        image = Image.open(image_path).convert('RGB')

        # Plot Gambar di subplot bawah
        ax = plt.subplot2grid((rows + 1, cols), (1 + i // cols, i % cols))  # Baris berikutnya untuk gambar
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Actual: {actual_label_name}\nPredicted: {predicted_label_name}", fontsize=8)  # Perkecil font
    except Exception as e:
        print(f"Error processing image {i}: {e}")

# Atur layout dan tampilkan
plt.tight_layout()
plt.show()