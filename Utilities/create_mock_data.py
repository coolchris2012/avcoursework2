import numpy as np
from pathlib import Path

# Simple mock visual features - matches MFCC format exactly
# Audio: (frames, 12), Visual: (frames, 30)

audio_dir = Path("Audio_MFCCs")
visual_dir = Path("Visual_Features_Mock")
visual_dir.mkdir(parents=True, exist_ok=True)

for audio_file in audio_dir.glob("*.npy"):
    mfcc = np.load(audio_file)  # Load existing audio
    num_frames = mfcc.shape[0]
    
    # Create visual features: same frames, 30 dimensions
    visual = np.random.randn(num_frames, 30).astype(np.float32)
    
    np.save(visual_dir / audio_file.name, visual)

print(f"Created {len(list(visual_dir.glob('*.npy')))} mock visual files")
