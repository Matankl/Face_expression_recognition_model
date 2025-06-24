import torch
import numpy as np
from torch.utils.data import DataLoader

def calculate_class_weights(dataloader: DataLoader) -> torch.Tensor:
    """
    Calculate class weights from dataloader.
    Args:
        dataloader: Dataloader that contains the training data
    Returns:
        torch.Tensor: Weights for each class
    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    class_counts = np.bincount(all_labels)
    
    n_samples = len(all_labels)
    
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts)
    
    weights = torch.FloatTensor(weights)
    
    return weights
