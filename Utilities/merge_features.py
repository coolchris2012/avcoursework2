# CNN/Utils/merge_features.py

import numpy as np
from pathlib import Path

audio_dir = Path("Audio_MFCCs")
visual_dir = Path("Visual_Features_Mock")
output_dir = Path("Merged_Features_Mock")
output_dir.mkdir(exist_ok=True)

for audio_file in audio_dir.glob("*.npy"):
    visual_file = visual_dir / audio_file.name # Find the matching visual file by name
    
    if not visual_file.exists():
        continue # Ignore if there isnt a matching visual file
    
    audio = np.load(audio_file)   # (frames, MFCC coeffs, e.g., 12)
    visual = np.load(visual_file)  # (frames, DCT coeffs, e.g., 30)
    
    # Concatenate along feature dimension, adding more possible features for the same number of frames
    merged = np.concatenate([audio, visual], axis=1)  # (frames, 42)
    
    np.save(output_dir / audio_file.name, merged)

print(f"Created {len(list(output_dir.glob('*.npy')))} merged files")