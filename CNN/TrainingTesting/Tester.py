from CNN.Settings import *


class Tester:
    """
    Handles evaluation of the trained speech recognition model.
    Calculates accuracy, generates confusion matrix, and provides detailed metrics.
    """
    
    def __init__(self, model, test_loader, device, class_names):
        """
        Initialize tester.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test data
            device: torch.device to use for evaluation
            class_names: List of class names in order
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Loss function
        self.criterion = config.LOSS_FUNCTION()
        
        # Results storage
        self.all_predictions = []
        self.all_labels = []
        self.all_outputs = []
    

#======================FULL EVALUATION PROCES======================

    
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            results: Dictionary containing test loss, accuracy, and detailed metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Reset storage
        self.all_predictions = []
        self.all_labels = []
        self.all_outputs = []
        
        print(f"\nEvaluating on {len(self.test_loader.dataset)} test samples...")
        
        with torch.no_grad():
            for mfccs, labels in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                mfccs, labels = mfccs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(mfccs)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                _, predicted = torch.max(outputs, dim=1)
                
                # Track metrics
                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Store for confusion matrix
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_outputs.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        results = {
            'test_loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        return results
    

#======================METRIC CALCULATION METHODS======================


    def confusion_matrix(self):
        """
        Generate confusion matrix from stored predictions. \n
        This maatrix helps visualize the performance of the classification model by
        showing the counts of true vs predicted labels. And can pinpoint exactly which
        classes are being confused by the model.
        
        Returns:
            conf_matrix: numpy array of shape (num_classes, num_classes)
        """
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for true_label, pred_label in zip(self.all_labels, self.all_predictions):
            conf_matrix[true_label][pred_label] += 1
        
        return conf_matrix
    
    def per_class_accuracy(self):
        """
        Calculate accuracy for each class.
        
        Returns:
            per_class_acc: Dictionary mapping class names to their accuracies
        """
        conf_matrix = self.confusion_matrix()
        per_class_acc = {}
        
        for i, class_name in enumerate(self.class_names):
            # Correct predictions for this class
            correct = conf_matrix[i][i]
            # Total samples of this class
            total = conf_matrix[i].sum()
            
            if total > 0:
                accuracy = 100 * correct / total
            else:
                accuracy = 0.0
            
            per_class_acc[class_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        
        return per_class_acc
    

#=================================PRINTING METHODS=================================


    def print_results(self, results):
        """
        Print formatted test results.
        
        Args:
            results: Dictionary from evaluate()
        """
        print(f"\n{'='*60}")
        print(f"Test Results")
        print(f"{'='*60}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print(f"{'='*60}\n")
    
    def print_confusion_matrix(self):
        """
        Print formatted confusion matrix.
        """
        conf_matrix = self.confusion_matrix()
        
        print(f"\n{'='*60}")
        print(f"Confusion Matrix")
        print(f"{'='*60}")
        print(f"{'True \\ Pred':<15}", end="")
        
        # Print column headers (abbreviated)
        for name in self.class_names:
            print(f"{name[:8]:<10}", end="")
        print()
        
        # Print rows
        for i, true_class in enumerate(self.class_names):
            print(f"{true_class[:12]:<15}", end="")
            for j in range(self.num_classes):
                print(f"{conf_matrix[i][j]:<10}", end="")
            print()
        print()
    
    def print_per_class_accuracy(self):
        """
        Print per-class accuracy statistics.
        """
        per_class_acc = self.per_class_accuracy()
        
        print(f"\n{'='*60}")
        print(f"Per-Class Accuracy")
        print(f"{'='*60}")
        print(f"{'Class':<15} {'Accuracy':<12} {'Correct/Total':<15}")
        print(f"{'-'*60}")
        
        for class_name, metrics in per_class_acc.items():
            acc = metrics['accuracy']
            correct = metrics['correct']
            total = metrics['total']
            print(f"{class_name:<15} {acc:>6.2f}%      {correct:>3}/{total:<3}")
        
        print(f"{'='*60}\n")
    


#=================================PLOTTING METHODS=================================


    
    def plot_confusion_matrix(self, save_path, show=True):
        """
        Plot confusion matrix as a simple grid.
        
        Args:
            save_path: Path to save the plot (default: RUN_FOLDER/confusion_matrix.png)
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        conf_matrix = self.confusion_matrix()
        
        # Create simple plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display matrix as image
        im = ax.imshow(conf_matrix, cmap='Blues')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set ticks
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(self.class_names)
        
        # Add text annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                text = ax.text(j, i, conf_matrix[i, j],
                             ha="center", va="center", color="black")
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_per_class_accuracy(self, save_path, show=True):
        """
        Plot per-class accuracy as a simple bar chart.
        
        Args:
            save_path: Path to save the plot (default: RUN_FOLDER/per_class_accuracy.png)
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        per_class_acc = self.per_class_accuracy()
        
        # Extract data for plotting
        classes = list(per_class_acc.keys())
        accuracies = [per_class_acc[c]['accuracy'] for c in classes]
        
        # Create simple bar chart
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(classes)), accuracies)
        
        plt.xlabel('Class Name')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_all_metrics(self, save_dir, show=False):
        """
        Generate all visualization plots.
        
        Args:
            save_dir: Directory to save plots (default: RUN_FOLDER)
            show: Whether to display the plots
        """
        
        print(f"\nGenerating visualization plots...")
        
        # Confusion matrix heatmap
        self.plot_confusion_matrix(
            save_path=save_dir / "confusion_matrix.png",
            show=show
        )
        
        # Per-class accuracy bar chart
        self.plot_per_class_accuracy(
            save_path=save_dir / "per_class_accuracy.png",
            show=show
        )
        
        print(f"All plots saved to: {save_dir}\n")
