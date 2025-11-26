"""
Live Speech Recognition Demo
Records audio from microphone, extracts MFCCs, and predicts the spoken name.
"""

from datetime import datetime
import torch
import numpy as np
import sounddevice as sd
import json
from pathlib import Path
import sys
import shutil

# Add parent directory to path to import featureExtraction
sys.path.append(str(Path(__file__).parent.parent))

from CNN.Models.ModelArchitecture import SpeechClassifier
import featureExtraction






# Load model and class mappings
def load_model_and_mappings(model_path):
    """Load trained model and class mappings"""
    
    model_dir = Path(model_path).parent
    
    # Load class mappings
    with open(model_dir / "class_mappings.json", 'r') as f:
        mappings = json.load(f)
    
    # Create model
    model = SpeechClassifier(num_classes=20)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model state dict if checkpoint contains additional info
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, mappings['idx_to_class']

# Prediction func
def predict_name(model, mfcc, idx_to_class):
    """Run inference and return predicted name"""
    
    # Convert to tensor and add batch dimension: (1, 1, 20, 12)
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        logits = model(mfcc_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx_tensor = torch.argmax(probabilities, dim=1)
        predicted_idx = int(predicted_idx_tensor.item())
        probs_flat = probabilities[0]
        confidence = float(probs_flat[predicted_idx].item())
    
    predicted_name = idx_to_class[str(predicted_idx)]
    
    return predicted_name, confidence, probs_flat


# Main Loop func
def main():
    # Config
    MODEL_PATH = Path("CNN/Models/BaseResults/best_model.pth")
    SAMPLE_RATE = 16000
    RECORDING_SECONDS = 2
    
    print("============================================================")
    print("LIVE SPEECH RECOGNITION DEMO")
    print("============================================================")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model, idx_to_class = load_model_and_mappings(MODEL_PATH)
    print("Model loaded successfully")
    
    while True:
        print("\n ============================================================")
        input("Press ENTER to start recording")
        
        # Record audio
        print(f"Recording for {RECORDING_SECONDS} seconds...")
        signal = sd.rec(RECORDING_SECONDS * SAMPLE_RATE, 
                       samplerate=SAMPLE_RATE, 
                       channels=1,
                       dtype='float32')
        sd.wait()
        print("Recording complete")
        
        # Play back
        print("Playing back...")
        sd.play(signal, SAMPLE_RATE)
        sd.wait()
        
        # Extract MFCCs using featureExtraction.main
        print("Extracting MFCC features")
        signal_flat = signal.flatten()
        folderName = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        featureExtraction.main(signal_flat, SAMPLE_RATE, "temp_mfcc", folderName)
        
        # Load the saved MFCC
        mfcc = np.load(f"{folderName}/temp_mfcc.npy")
        print(f"MFCC shape: {mfcc.shape}")
        
        # Predict
        print("Creating forward pass")
        predicted_name, confidence, all_probs = predict_name(model, mfcc, idx_to_class)
        
        # Display results
        print("\n ============================================================")
        print(f"PREDICTION: {predicted_name.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("============================================================")
        
        # Show top 3 predictions
        print("\nTop 3 predictions:")
        top3_indices = torch.topk(all_probs, 3).indices
        for i, idx in enumerate(top3_indices, 1):
            name = idx_to_class[str(idx.item())]
            prob = all_probs[idx].item()
            print(f"  {i}. {name:15s} - {prob * 100:5.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Demo ended.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp folder
        for folder in Path(".").glob("????-??-??_??-??-??"):
            if folder.is_dir():
                shutil.rmtree(folder)
