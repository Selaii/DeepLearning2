import torch
from torch.utils.data import DataLoader
from config import set_dataset_paths, BATCH_SIZE
from prepare_dataset import ImageTensorDataset
from models.model_ann import ANNModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load Dataset
_, test_dataset_path = set_dataset_paths()
test_dataset = ImageTensorDataset(test_dataset_path)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
input_size = 224 * 224 * 3
num_classes = len(test_dataset.class_to_idx)
model = ANNModel(input_size, num_classes)
model.load_state_dict(torch.load("models/model_ann.pth"))
model.eval()

# Evaluation
def evaluate():
    all_labels = []
    all_preds = []
    all_image_paths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
            all_image_paths.extend(test_dataset.image_paths[i] for i in range(len(labels)))

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save results
    results_df = pd.DataFrame({"Actual Label": all_labels, "Predicted Label": all_preds, "Image Path": all_image_paths})
    results_df.to_csv("results/test_results.csv", index=False)
    print("Hasil evaluasi disimpan di results/test_results.csv")

if __name__ == "__main__":
    evaluate()