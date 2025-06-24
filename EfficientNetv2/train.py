import time
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from ema_pytorch import EMA
from torchinfo import summary
import torch.nn.functional as F
from efficientfer import EfficientFER
from torch.amp import GradScaler, autocast
from dataloader import get_train_dataloaders
from calculate_weight import calculate_class_weights

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Configuration parameters
CONFIG = {
    # Dataset parameters
    'data_dir': r'C:\Users\matan\Desktop\Code\DataSets\Face_expression_recognition',
    'num_classes': 7,
    'batch_size': 64,
    'num_workers': 20,
    'image_size': 224,

    # Training parameters
    'epochs': 50,
    'patience': 7,
    'learning_rate': 5e-3,
    'weight_decay': 0.03,
    'min_lr': 1e-6,
    
    # Checkpoint parameters
    'checkpoint_dir': 'checkpoints',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Result file
    'result_file': 'result.txt',
}

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)  
        ce_loss = -log_prob.gather(1, target.unsqueeze(1)).squeeze()
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[target]
            
        return focal_loss.mean()
       
def setup_training():
    """Prepare training components"""
    # Create checkpoint directory
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # EfficientFER
    model = EfficientFER(
        num_classes=CONFIG['num_classes'],
    ).to(CONFIG['device'])
    
    # EMA
    ema_model = EMA(
        model,
        beta = 0.999,          
        update_after_step = 100,
        update_every = 1,   
    )

    # Model summary
    print("\nModel Summary:")
    model_summary = summary(
        model,
        input_size=(1, 3, CONFIG['image_size'], CONFIG['image_size']),
        col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"]
    )

    # Get dataloaders
    train_loader, val_loader = get_train_dataloaders(
        CONFIG['data_dir'],
        CONFIG['batch_size'],
        CONFIG['num_workers'],
        CONFIG['image_size']
    )
    print(f'Number of Training Samples: {len(train_loader.dataset)}')
    print(f'Number of Validation Samples: {len(val_loader.dataset)}')
    print("=============================================================================================================================")

    # Calculate class weights
    class_weights = calculate_class_weights(train_loader).to(CONFIG['device'])
    print(f'Class Weights: {class_weights}')
    print("=============================================================================================================================")

    # Loss function
    criterion = FocalLoss(weight=class_weights, gamma=2.0, reduction='mean')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
   
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=CONFIG['epochs'] * len(train_loader),
        pct_start=0.3
    )

    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    return model, ema_model, train_loader, val_loader, criterion, optimizer, scheduler, scaler

def train_epoch(model, ema_model, train_loader, criterion, optimizer, scaler, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG["epochs"]} [Train]')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            main_loss = criterion(outputs, targets)
            loss = main_loss
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=1.0,
            error_if_nonfinite=True
        )
        
        scaler.step(optimizer)
        scaler.update()
        
        # EMA güncelleme
        ema_model.update()
        
        optimizer.zero_grad()
        
        if torch.isnan(loss):
            print(f"\nNaN loss detected at batch {batch_idx}!")
            continue
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Progress bar güncelleme
        avg_loss = total_loss / (batch_idx + 1)
        acc = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%'
        })
    
    return total_loss / len(train_loader), acc

def validate(model, val_loader, criterion):
    """Evaluate the model on the validation set"""
    eval_model = model.ema_model 
    eval_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = eval_model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
    
    return total_loss / len(val_loader), acc

def save_checkpoint(model, ema_model, optimizer, scheduler, epoch, best_val_acc, is_best=False):
    """Save the model checkpoint"""
    if is_best:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'config': CONFIG
        }
        best_model_path = Path(CONFIG['checkpoint_dir']) / 'best_model.pth'
        torch.save(checkpoint, best_model_path)
        print(f"\nNew best model saved! (Epoch {epoch}, Val Acc: {best_val_acc:.2f}%)")

def log_results(epoch, train_loss, train_acc, val_loss, val_acc):
    """Save training results to file"""
    with open(CONFIG['result_file'], 'a') as f:
        f.write(f"Epoch: {epoch}/{CONFIG['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%\n")

def main():
    """Main training function"""
    print("Training started...")
    print(f"Using device: {CONFIG['device']}")
    
    # Prepare training components
    model, ema_model, train_loader, val_loader, criterion, optimizer, scheduler, scaler = setup_training()
    
    # Variables for tracking training
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Initialize result file
    with open(CONFIG['result_file'], 'w') as f:
        f.write("")
    
    # Training loop
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        
        # Training
        train_loss, train_acc = train_epoch(model, ema_model, train_loader, criterion, optimizer, scaler, epoch)
        
        # Validation with only EMA model
        val_loss, val_acc = validate(ema_model, val_loader, criterion)
        
        # Save results
        log_results(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Save the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, ema_model, optimizer, scheduler, epoch, best_val_acc, is_best)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping control
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
