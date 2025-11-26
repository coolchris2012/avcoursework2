from CNN.Settings import *


class Trainer:
    """
    Handles the training loop for the speech recognition model.
    Focuses purely on training mechanics - dataset creation and evaluation 
    are handled separately.
    """
    
    def __init__(self, model, train_loader, device):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            device: torch.device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = config.LOSS_FUNCTION()
        self.optimizer = config.OPTIMIZER(
            model.parameters(),
            **config.OPTIMIZER_PARAMS
        )
        
        # Learning rate scheduler
        self.scheduler = config.LR_SCHEDULER(
            self.optimizer,
            **config.LR_SCHEDULER_PARAMS
        )
        
        # Training state
        self.current_epoch = 0
        self.best_train_loss = float('inf')
        self.history = {
            'train_loss': []
        }
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for mfccs, labels in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{config.NUM_EPOCHS}"):
            # Move data to device
            mfccs, labels = mfccs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(mfccs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()  # Clear gradients from previous batch
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs=None, save_path=None):
        """
        Run the full training loop.
        
        Args:
            num_epochs: Number of epochs to train (default: from config)
            save_path: Where to save the best model (default: from config)
        
        Returns:
            history: Dictionary containing training history
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS
        if save_path is None:
            save_path = config.MODEL_SAVE_PATH
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Update learning rate based on training loss
            self.scheduler.step(train_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            
            # Print results
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Save best model
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                
                # Create directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                }, save_path)
                print(f"  ✓ Best model saved (Train Loss: {train_loss:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best Train Loss: {self.best_train_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def plot_training_history(self, save_path=None, show=True):
        """
        Plot training loss curve.
        
        Args:
            save_path: Path to save the plot (default: RUN_FOLDER/training_loss.png)
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        
        if save_path is None:
            save_path = config.RUN_FOLDER / "training_loss.png"
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Simple plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_loss'])
        
        # Mark best epoch
        best_epoch = self.history['train_loss'].index(self.best_train_loss) + 1
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best (Epoch {best_epoch})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training loss plot saved to: {save_path}")
        
        if show:
            plt.show()
        plt.close()
