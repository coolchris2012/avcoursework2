from pathlib import Path
import featureExtraction
import soundfile as sf

# Generates MFCC values for all files in Raw_Clips

directory = Path("Raw_Clips/audio")
files = directory.glob("*.wav")
for filename in files:
  signal, sampleRate = sf.read("Raw_Clips/audio/" + filename.name, dtype='float32')
  # Convert stereo to mono if needed
  if signal.ndim == 2:
    signal = signal.mean(axis=1)
  name = filename.name.replace(".wav", "")
  featureExtraction.main(signal, sampleRate, name, "MFCCs_00") # Manually iterate per run