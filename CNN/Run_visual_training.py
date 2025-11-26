import os
import sys
from pathlib import Path

# Change to AVCoursework2 directory if running directly
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent  # CNN/
    project_root = script_dir.parent  # AVCoursework2/
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    print(f"Working directory set to: {os.getcwd()}\n")

from CNN.Settings import *
from CNN.Dataset import *
from CNN.Models import *
from CNN.TrainingTesting import *


def main():
    """
    Main entry point for training the speech recognition model.
    Handles dataset creation, model initialization, and training orchestration.
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset and create train/validation split
    data_dir = config.Visual_MFFCs_DIR
    print(f"\nLoading dataset from {data_dir}...")
    train_dataset, test_dataset = VisualDataset.from_split(
        data_dir=data_dir,
        train_transform=config.TRAIN_TRANSFORMATION,
        test_transform=config.TEST_TRANSFORMATION,
        split_ratio=config.TRAIN_SPLIT_RATIO,
        seed=config.RANDOM_SEED
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = train_dataset.create_dataloader(shuffle=True, batch_size=config.BATCH_SIZE)
    test_loader = test_dataset.create_dataloader(shuffle=False, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    print(f"\nInitializing SpeechClassifier...")
    model = SpeechClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_units=config.HIDDEN_UNITS
    ).to(device)
    
    # Print model summary
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device
    )
    
    # Train the model
    history = trainer.train(save_path=config.VISUAL_SAVE_PATH)
    
    # Plot training history
    trainer.plot_training_history(VISUAL_SAVE_FOLDER, show=False)
    
    # Save class mappings to the timestamped run folder
    class_mapping_path = config.VISUAL_SAVE_FOLDER / "class_mappings.json"
    import json
    with open(class_mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': train_dataset.class_to_idx,
            'idx_to_class': {v: k for k, v in train_dataset.class_to_idx.items()}
        }, f, indent=2)
    print(f"Class mappings saved to: {class_mapping_path}")
    
    # Load best model for testing
    print(f"\nLoading best model for evaluation...")
    checkpoint = torch.load(config.VISUAL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model loaded from epoch {checkpoint['epoch']+1}")
    
    # Test the model
    tester = Tester(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=train_dataset.classes
    )
    
    # Evaluate and print results
    results = tester.evaluate()
    tester.print_results(results)
    tester.print_per_class_accuracy()
    
    # Save test results summary
    results_path = config.VISUAL_SAVE_FOLDER / "test_summary.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'test_loss': results['test_loss'],
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total']
        }, f, indent=2)
    print(f"Test summary saved to: {results_path}")
    
    # Generate visualization plots (confusion matrix + per-class accuracy)
    tester.plot_all_metrics(save_dir=config.VISUAL_SAVE_FOLDER, show=False)


if __name__ == "__main__":
    main()