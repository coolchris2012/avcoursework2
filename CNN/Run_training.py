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



    
    # ================================== SELECT DATASET BASED ON MODALITY ==================================

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
    




    # ================================== MODEL INITIALISATION ==================================
    

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





    # ================================== TRAINING PROCESS ==================================
    
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






    # ================================== TESTING PROCESS ==================================
    
    # Load best model for testing
    print(f"\nLoading best model for evaluation...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model loaded from epoch {checkpoint['epoch']+1}")
    
    # Determine SNR levels to test
    if modality in ['audio', 'early_fusion']:
        snr_levels = [(None, "CleanTest"), (20, "NoisyTest_20dB"), (10, "NoisyTest_10dB"), 
                      (0, "NoisyTest_0dB"), (-5, "NoisyTest_-5dB")]
    else:  # visual - no noise needed
        snr_levels = [(None, "")]
    
    for snr_db, folder_name in snr_levels:
        # Set up noise for this test
        config.ADD_NOISE = snr_db is not None
        if snr_db is not None:
            config.SNR_DB = snr_db
        
        # Create results folder
        test_folder = save_folder / folder_name if folder_name else save_folder
        test_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTesting at: {'CLEAN' if snr_db is None else f'{snr_db}dB SNR'}")
        
        # Reload dataset with current noise settings
        if modality == 'audio':
            _, test_dataset = AudioDataset.from_split(audio_dir or config.MFCC_DIR, 
                test_transform=config.TEST_TRANSFORMATION, split_ratio=config.TRAIN_SPLIT_RATIO, seed=config.RANDOM_SEED)
        elif modality == 'visual':
            _, test_dataset = VisualDataset.from_split(visual_dir or config.VISUAL_FEATURES_DIR,
                test_transform=config.TEST_TRANSFORMATION, split_ratio=config.TRAIN_SPLIT_RATIO, seed=config.RANDOM_SEED)
        elif modality == 'early_fusion':
            _, test_dataset = AudioVisualDataset.from_split(audio_dir or config.MFCC_DIR, visual_dir or config.VISUAL_FEATURES_DIR,
                test_transform=config.TEST_TRANSFORMATION, split_ratio=config.TRAIN_SPLIT_RATIO, seed=config.RANDOM_SEED)
        
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                                 num_workers=config.NUM_WORKERS, collate_fn=test_dataset.collate_fn)
        
        # Test and save results
        tester = Tester(model, test_loader, device, test_dataset.classes)
        results = tester.evaluate()
        tester.print_results(results)
        tester.print_per_class_accuracy()
        
        with open(test_folder / "test_summary.json", 'w') as f:
            json.dump({'modality': modality, 'snr_db': snr_db, 'test_loss': results['test_loss'],
                      'accuracy': results['accuracy'], 'correct': results['correct'], 'total': results['total']}, f, indent=2)
        
        tester.plot_all_metrics(save_dir=test_folder, show=False)
    
    print(f"\n{'='*60}")
    print(f"Training complete for {modality.upper()} model!")
    print(f"Model saved to: {save_path}")
    print(f"Results saved to: {save_folder}")
    print(f"{'='*60}\n")


# Run the main function
if __name__ == "__main__":

    print(f"Working directory: {os.getcwd()}\n")
    
    train_model(
        modality="early_fusion",
        audio_dir=config.MFCC_DIR,
        visual_dir=config.VISUAL_FEATURES_DIR,
    ) 