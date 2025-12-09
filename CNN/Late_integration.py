"""
Late Fusion Integration for Audio-Visual Speech Recognition
Loads pre-trained audio and visual models, combines their predictions
"""

import os
import sys
from pathlib import Path

# Set working directory to project root (AVCoursework2/)
script_dir = Path(__file__).resolve().parent  # CNN/
project_root = script_dir.parent  # AVCoursework2/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import others after path setup
import torch
import numpy as np
from torch.utils.data import DataLoader
from CNN.Settings import config
from CNN.Dataset import *
from CNN.Models.ModelArchitecture import SpeechClassifier
from CNN.TrainingTesting.Tester import Tester


class LateFusionTester:
    def __init__(self, audio_model_path, visual_model_path, device='cuda'):
        """
        Initialize late fusion with pre-trained models
        
        Args:
            audio_model_path: Path to trained audio model (best_model.pth)
            visual_model_path: Path to trained visual model (best_model.pth)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load audio model
        audio_checkpoint = torch.load(audio_model_path, map_location=device)
        self.audio_model = SpeechClassifier(
            num_classes=config.NUM_CLASSES,
            hidden_units=config.HIDDEN_UNITS
        ).to(device)
        self.audio_model.load_state_dict(audio_checkpoint['model_state_dict'])
        self.audio_model.eval()
        
        # Load visual model
        visual_checkpoint = torch.load(visual_model_path, map_location=device)
        self.visual_model = SpeechClassifier(
            num_classes=config.NUM_CLASSES,
            hidden_units=config.HIDDEN_UNITS
        ).to(device)
        self.visual_model.load_state_dict(visual_checkpoint['model_state_dict'])
        self.visual_model.eval()
        
        print(f"Loaded audio model from {audio_model_path}")
        print(f"Loaded visual model from {visual_model_path}")
    
    def test(self, audio_dataset, visual_dataset, batch_size=32, fusion_method='average'):
        """
        Test late fusion on datasets
        
        Args:
            audio_dataset: AudioDataset for testing
            visual_dataset: VisualDataset for testing (must be same samples as audio)
            batch_size: Batch size for testing
            fusion_method: 'average' (probability averaging) or 'voting' (hard voting)
        
        Returns:
            accuracy, all_predictions, all_labels
        """
        audio_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, 
                                 collate_fn=audio_dataset.collate_fn)
        visual_loader = DataLoader(visual_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=visual_dataset.collate_fn)
        
        all_audio_probs = []
        all_visual_probs = []
        all_labels = []
        
        with torch.no_grad():
            for (audio_data, audio_labels), (visual_data, visual_labels) in zip(audio_loader, visual_loader):
                # Verify labels match (sanity check)
                assert torch.all(audio_labels == visual_labels), "Audio and visual labels must match!"
                
                # Get predictions from both models
                audio_data = audio_data.to(self.device)
                visual_data = visual_data.to(self.device)
                
                audio_outputs = self.audio_model(audio_data)
                visual_outputs = self.visual_model(visual_data)
                
                # Get probabilities
                audio_probs = torch.softmax(audio_outputs, dim=1)
                visual_probs = torch.softmax(visual_outputs, dim=1)
                
                all_audio_probs.append(audio_probs.cpu())
                all_visual_probs.append(visual_probs.cpu())
                all_labels.append(audio_labels)
        
        # Concatenate all batches
        all_audio_probs = torch.cat(all_audio_probs, dim=0)
        all_visual_probs = torch.cat(all_visual_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Fusion
        if fusion_method == 'average':
            # Average probability fusion
            fused_probs = (all_audio_probs + all_visual_probs) / 2.0
            predictions = torch.argmax(fused_probs, dim=1)
        elif fusion_method == 'voting':
            # Hard voting
            audio_preds = torch.argmax(all_audio_probs, dim=1)
            visual_preds = torch.argmax(all_visual_probs, dim=1)
            # If they agree, use that; if not, use audio (or could use visual)
            predictions = torch.where(audio_preds == visual_preds, audio_preds, audio_preds)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Calculate accuracy
        correct = (predictions == all_labels).sum().item()
        accuracy = correct / len(all_labels)
        
        return accuracy, predictions.numpy(), all_labels.numpy()


def test_late_fusion_at_snr_levels():
    """
    Test late fusion at multiple SNR levels
    Uses clean-trained audio and visual models
    """
    # Paths to clean-trained models
    audio_model_path = Path("CNN/TrainedModels/Audio/BaseModel/best_model.pth")
    visual_model_path = Path("CNN/TrainedModels/Visual/BaseModel/best_model.pth")
    
    if not audio_model_path.exists():
        print(f"Audio model not found at {audio_model_path}")
        return
    
    if not visual_model_path.exists():
        print(f"Visual model not found at {visual_model_path}")
        return
    
    # Initialize tester
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = LateFusionTester(audio_model_path, visual_model_path, device=device)
    
    # Test at multiple SNR levels
    snr_levels = [
        (None, "CleanTest"),
        (20, "NoisyTest_20dB"),
        (10, "NoisyTest_10dB"),
        (0, "NoisyTest_0dB"),
        (-5, "NoisyTest_-5dB")
    ]
    
    results = {}
    
    for snr_db, folder_name in snr_levels:
        print(f"\n{'='*60}")
        print(f"Testing at SNR: {snr_db if snr_db is not None else 'Clean'}")
        print(f"{'='*60}")
        
        # Configure noise for audio dataset
        original_add_noise = config.ADD_NOISE
        original_snr = config.SNR_DB
        
        if snr_db is not None:
            config.ADD_NOISE = True
            config.SNR_DB = snr_db
        else:
            config.ADD_NOISE = False
        
        # Create test datasets
        audio_test_dataset = AudioDataset(config.MFCC_DIR)
        visual_test_dataset = VisualDataset(config.VISUAL_FEATURES_DIR)
        
        # Test
        accuracy, predictions, labels = tester.test(
            audio_test_dataset,
            visual_test_dataset,
            batch_size=32,
            fusion_method='average'
        )
        
        results[folder_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Restore config
        config.ADD_NOISE = original_add_noise
        config.SNR_DB = original_snr
        
        # Save results
        output_dir = Path("CNN/TrainedModels/LateFusion/BaseModel") / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "labels.npy", labels)
        
        # Create a dummy Tester instance to use its plotting methods
        dummy_tester = Tester(None, None, None, audio_test_dataset.classes)
        dummy_tester.all_predictions = predictions
        dummy_tester.all_labels = labels
        dummy_tester.plot_confusion_matrix(output_dir / "confusion_matrix.png", show=False)
        dummy_tester.plot_per_class_accuracy(output_dir / "per_class_accuracy.png", show=False)
            f.write(f"Late Fusion Testing Results\n")
            f.write(f"SNR: {snr_db if snr_db is not None else 'Clean'}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Fusion Method: Average\n")
        
        print(f"Results saved to {output_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Late Fusion Summary")
    print(f"{'='*60}")
    for folder_name, result in results.items():
        print(f"{folder_name}: {result['accuracy']:.4f}")


if __name__ == "__main__":
    test_late_fusion_at_snr_levels()
