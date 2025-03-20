import matplotlib.pyplot as plt
import numpy as np
import torch

# Plot Training Loss
def plot_training_loss(training_errors, output_path="results/training_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_errors) + 1), training_errors, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Error over Epochs")
    plt.grid()
    plt.savefig(output_path)
    print(f"Training loss plot disimpan di {output_path}")
    plt.show()

# Plot Sample Predictions
def plot_sample_predictions(model, dataset, class_to_idx, device, num_images=5):
    model.eval()
    class_names = {v: k for k, v in class_to_idx.items()}
    fig, axes = plt.subplots(1, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        img_tensor, label = dataset[i]
        img = img_tensor.permute(1, 2, 0).numpy()
        
        # Predict
        img_tensor = img_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
        
        pred_label = class_names[predicted.item()]
        true_label = class_names[label]

        # Display Image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Actual: {true_label}\nPredicted: {pred_label}")
    plt.tight_layout()
    plt.show()

# Save Model
def save_model(model, path="models/model_ann.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model berhasil disimpan di {path}")

# Load Model
def load_model(model, path="models/model_ann.pth"):
    model.load_state_dict(torch.load(path))
    print(f"Model berhasil dimuat dari {path}")
    return model
