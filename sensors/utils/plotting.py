import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_confusion_matrix_png(confusion_matrix, epoch, save_dir, split="train", class_names=None):
    """Save confusion matrix as PNG file for each epoch"""
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(confusion_matrix.shape[0])]

    # Create DataFrame for better visualization
    cm_normalized = confusion_matrix.astype(float)
    col_sums = cm_normalized.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_normalized = cm_normalized / col_sums
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    # Create the plot
    plt.figure(figsize=(10, 8))
    cfm_plot = sns.heatmap(df_cm, annot=True, fmt='.3f', cmap='viridis',
                           cbar=True, square=True)
    plt.title(f'Confusion Matrix - {split.capitalize()} - Epoch {epoch + 1}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save the plot
    filename = f'confusion_matrix_{split}_epoch_{epoch + 1}.png'
    filepath = save_dir / filename
    cfm_plot.figure.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Important: close the figure to free memory

    return filepath
