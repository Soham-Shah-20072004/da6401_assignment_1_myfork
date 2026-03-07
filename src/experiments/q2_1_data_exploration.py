"""
Q2.1 Data Exploration (3 Marks)
Logs 5 sample images from each of the 10 classes in the MNIST dataset to a Weights & Biases Table.
"""

import sys
import os
import numpy as np
import wandb

# Add src to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def log_visual_similarity(X_train, y_train):
    # 1. Compute mean image per class (shape: 10 x 784)
    class_means = np.array([
        X_train[y_train == i].mean(axis=0) for i in range(10)
    ])
    
    # 2. Cosine similarity between every pair of class means
    sim_matrix = cosine_similarity(class_means)  # shape: 10x10
    
    # 3. Plot as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        annot=True, fmt=".2f",
        xticklabels=range(10),
        yticklabels=range(10),
        cmap="YlOrRd", ax=ax
    )
    ax.set_title("Pairwise Cosine Similarity of Class Mean Images")
    
    wandb.log({"Class_Similarity_Heatmap": wandb.Image(fig)})
    plt.close(fig)
    
    # 4. Also log the mean images themselves so you can visually inspect them
    mean_table = wandb.Table(columns=["Class", "Mean Image"])
    for i in range(10):
        img = class_means[i].reshape(28, 28)
        mean_table.add_data(i, wandb.Image(img, caption=f"Mean of class {i}"))
    wandb.log({"Class_Mean_Images": mean_table})


def main():
    print("Initializing W&B Run for Data Exploration...")
    wandb.init(project="da6401_assignment_1_myfork-src", name="q2.1-data-exploration", job_type="exploration")
    
    print("Loading MNIST dataset...")
    X_train, y_train, _, _ = load_data('mnist')
    
    class_names = [str(i) for i in range(10)]
    columns = ["Image", "Label", "Class Name"]
    sample_table = wandb.Table(columns=columns)
    
    print("Sampling 5 images per class...")
    np.random.seed(42)  # ← moved OUTSIDE the loop (fix from earlier)
    for class_id in range(10):
        class_indices = np.where(y_train == class_id)[0]
        selected_indices = np.random.choice(class_indices, 5, replace=False)
        
        for idx in selected_indices:
            reshaped_image = X_train[idx].reshape((28, 28))
            wandb_image = wandb.Image(reshaped_image, caption=f"Class: {class_names[class_id]}")
            sample_table.add_data(wandb_image, class_id, class_names[class_id])
            
    wandb.log({"MNIST_Samples": sample_table})
    
    # ✅ NOW call the similarity analysis, before finish()
    print("Computing class similarity...")
    log_visual_similarity(X_train, y_train)
    
    wandb.finish()
    print("✅ Done! Check your 'da6401_assignment_1_myfork-src' project in W&B.")

if __name__ == "__main__":
    main()