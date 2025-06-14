import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the Cotton80Dataset from the previous code
from cotton80_dataset import Cotton80Dataset, get_default_transforms, create_dataloaders


class Cotton80Classifier:
    """Cotton80 Classification using TIMM ResNet50"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0.0
        self.start_epoch = 0
        
        # Create output directory
        self.output_dir = os.path.join(args.output_dir, f"resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        # Initialize model, data, optimizer
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        # Load checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
        
        print(f"Training on device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        # Create dataloaders using the Cotton80Dataset
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = create_dataloaders(
            root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            download=self.args.download,
            zip_url=self.args.zip_url
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Number of classes: {self.num_classes}")
    
    def setup_model(self):
        """Setup the ResNet50 model using TIMM"""
        print("Setting up ResNet50 model...")
        
        # Create ResNet50 model with pretrained weights
        self.model = timm.create_model(
            'resnet50',
            pretrained=self.args.pretrained,
            num_classes=self.num_classes,
            drop_rate=self.args.dropout,
            drop_path_rate=self.args.drop_path
        )
        print(f"Model number of classes: {self.model.num_classes}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Setup optimizer, scheduler, and loss function"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        
        # Optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        # Learning rate scheduler
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs
            )
        elif self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:  # plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=10, factor=0.1
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss/total_samples:.4f}',
                'Acc': f'{acc:.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/BatchAcc', acc, step)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return {'loss': epoch_loss, 'acc': epoch_acc.item()}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return {'loss': epoch_loss, 'acc': epoch_acc.item()}
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        print("Testing the model...")
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = running_loss / total_samples
        test_acc = running_corrects.double() / total_samples
        
        # Calculate per-class accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        with open(os.path.join(self.output_dir, 'test_classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        return {'loss': test_loss, 'acc': test_acc.item()}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'args': self.args
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"New best model saved with accuracy: {self.best_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(self.start_epoch, self.args.epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            if self.args.scheduler == 'plateau':
                self.scheduler.step(val_metrics['acc'])
            else:
                self.scheduler.step()
            
            # Log metrics
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['acc'])
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['acc'])
            
            # Tensorboard logging
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/EpochAcc', train_metrics['acc'], epoch)
            self.writer.add_scalar('Val/EpochLoss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/EpochAcc', val_metrics['acc'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.args.epochs} ({epoch_time:.2f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['acc']
            
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.args.early_stopping > 0:
                if len(val_accs) >= self.args.early_stopping:
                    if all(val_accs[-i] <= val_accs[-self.args.early_stopping] for i in range(1, self.args.early_stopping)):
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        # Plot training curves
        self.plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        
        # Test the best model
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
        
        test_metrics = self.test()
        
        # Save final results
        results = {
            'best_val_acc': self.best_acc,
            'test_acc': test_metrics['acc'],
            'test_loss': test_metrics['loss']
        }
        
        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_acc:.4f}")
        print(f"Test accuracy: {test_metrics['acc']:.4f}")
    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training curves"""
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-', label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Cotton80 Classification with ResNet50')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='./data', help='Root directory for dataset')
    parser.add_argument('--zip-url', type=str, default='<place_holder>', help='URL to download dataset')
    parser.add_argument('--download', action='store_true', help='Download dataset if not exists')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop-path', type=float, default=0.0, help='Drop path rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--early-stopping', type=int, default=0, help='Early stopping patience (0 to disable)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Create trainer and start training
    trainer = Cotton80Classifier(args)
    trainer.train()