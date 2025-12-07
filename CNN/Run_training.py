import os
import sys
from pathlib import Path


# Set working directory to project root (AVCoursework2/)
script_dir = Path(__file__).resolve().parent  # CNN/
project_root = script_dir.parent  # AVCoursework2/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import others after path setup
from CNN.Settings import *
from CNN.Dataset import *
from CNN.Models import *
from CNN.TrainingTesting import *


def train_model(modality: str, audio_dir: Path | None = None, visual_dir: Path | None = None):
    """
    Train a speech recognition model for the specified modality.
    
    Args:
        modality: Type of model to train ('audio', 'visual', or 'early_fusion')
        audio_dir: Path to audio features (required for 'audio' and 'early_fusion')
        visual_dir: Path to visual features (required for 'visual' and 'early_fusion')
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training modality: {modality.upper()}\n")
    
    # Select dataset and save paths based on modality

    # TRAINING AUDIO MODEL ONLY
    if modality == 'audio':
        if audio_dir is None:
            audio_dir = config.MFCC_DIR
        print(f"Loading audio dataset from {audio_dir}...")
        train_dataset, test_dataset = AudioDataset.from_split(
            data_dir=audio_dir,
            train_transform=config.TRAIN_TRANSFORMATION,
            test_transform=config.TEST_TRANSFORMATION,
            split_ratio=config.TRAIN_SPLIT_RATIO,
            seed=config.RANDOM_SEED
        )
        save_folder = config.AUDIO_SAVE_FOLDER
        save_path = config.AUDIO_SAVE_PATH
        

    # TRAINING VISUAL MODEL ONLY
    elif modality == 'visual':
        if visual_dir is None:
            visual_dir = config.VISUAL_FEATURES_DIR
        print(f"Loading visual dataset from {visual_dir}...")
        train_dataset, test_dataset = VisualDataset.from_split(
            data_dir=visual_dir,
            train_transform=config.TRAIN_TRANSFORMATION,
            test_transform=config.TEST_TRANSFORMATION,
            split_ratio=config.TRAIN_SPLIT_RATIO,
            seed=config.RANDOM_SEED
        )
        save_folder = config.VISUAL_SAVE_FOLDER
        save_path = config.VISUAL_SAVE_PATH
        

    # TRAINING EARLY FUSION AUDIO-VISUAL MODEL
    elif modality == 'early_fusion':
        if audio_dir is None:
            audio_dir = config.MFCC_DIR
        if visual_dir is None:
            visual_dir = config.VISUAL_FEATURES_DIR
        print(f"Loading audio-visual dataset from:")
        print(f"  Audio: {audio_dir}")
        print(f"  Visual: {visual_dir}")
        train_dataset, test_dataset = AudioVisualDataset.from_split(
            audio_dir,
            visual_dir,
            train_transform=config.TRAIN_TRANSFORMATION,
            test_transform=config.TEST_TRANSFORMATION,
            split_ratio=config.TRAIN_SPLIT_RATIO,
            seed=config.RANDOM_SEED
        )
        save_folder = config.EARLY_FUSION_SAVE_FOLDER
        save_path = config.EARLY_FUSION_SAVE_PATH
        

    # Invalid modality
    else:
        raise ValueError(f"Invalid modality: {modality}. Must be 'audio', 'visual', or 'early_fusion'")
    


    # ================================== TRAINING PROCESS ==================================

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = train_dataset.create_dataloader(shuffle=True, batch_size=config.BATCH_SIZE)
    test_loader = test_dataset.create_dataloader(shuffle=False, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    model = SpeechClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_units=config.HIDDEN_UNITS
    ).to(device)
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device
    )
    
    # Train the model
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    history = trainer.train(save_path=save_path)
    
    # Plot training history
    trainer.plot_training_history(save_folder, show=False)
    
    # Save class mappings to the timestamped run folder
    class_mapping_path = save_folder / "class_mappings.json"
    import json
    with open(class_mapping_path, 'w') as f:
        json.dump({
            'class_to_idx': train_dataset.class_to_idx,
            'idx_to_class': {v: k for k, v in train_dataset.class_to_idx.items()}
        }, f, indent=2)
    print(f"Class mappings saved to: {class_mapping_path}")
    
    # Load best model for testing
    print(f"\nLoading best model for evaluation...")
    checkpoint = torch.load(save_path, map_location=device)
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
    results_path = save_folder / "test_summary.json"
    with open(results_path, 'w') as f:
        json.dump({
            'modality': modality,
            'test_loss': results['test_loss'],
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total']
        }, f, indent=2)
    print(f"Test summary saved to: {results_path}")
    
    # Generate visualization plots (confusion matrix + per-class accuracy)
    tester.plot_all_metrics(save_dir=save_folder, show=False)
    
    print(f"\n{'='*60}")
    print(f"Training complete for {modality.upper()} model!")
    print(f"Model saved to: {save_path}")
    print(f"Results saved to: {save_folder}")
    print(f"{'='*60}\n")


# Run the main function
if __name__ == "__main__":

    print(f"Working directory: {os.getcwd()}\n")
    
    train_model(
        modality="visual",
        audio_dir=config.MFCC_DIR,
        visual_dir=config.VISUAL_FEATURES_DIR,
    )