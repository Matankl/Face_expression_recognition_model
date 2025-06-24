import torch
from dataloader import test_dataloader
from efficientfer import EfficientFER 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def test():
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model loading
    model = EfficientFER(num_classes=7)
    checkpoint_path = 'checkpoints/best_model.pth'
    
    try:
        # Model weights load
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
      
        state_dict = checkpoint['model_state_dict']
        
        # Check state dict keys
        first_key = list(state_dict.keys())[0]
       
        
        # Check model structure
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
       
        # Load state dict
        load_result = model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        print("\nModel loaded and eval mode activated.")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test data load
    test_loader = test_dataloader('fer2013_extended', 256, 12, image_size=224)
    
    # Predictions and real labels
    all_predictions = []
    all_labels = []
    
    # Test process
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Class-wise accuracy calculation
    class_correct = [0] * 7
    class_total = [0] * 7
    for i in range(len(all_labels)):
        label = all_labels[i]
        pred = all_predictions[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    # Class-wise accuracy print
    print("\nClass-wise accuracy:")
    for i in range(7):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Sınıf {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    # Confusion Matrix creation
    cm = confusion_matrix(all_labels, all_predictions)
    
    # General test accuracy calculation
    test_accuracy = np.mean(np.array(all_labels) == np.array(all_predictions)) * 100
    print(f"\nGeneral Test Accuracy: {test_accuracy:.4f}%")
    
    # Confusion Matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Real Values')
    plt.xlabel('Predicted Values')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Classification Report print
    report = classification_report(all_labels, all_predictions)
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    test()
