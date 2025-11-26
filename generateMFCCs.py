from pathlib import Path
import featureExtraction
import soundfile as sf

# Generates MFCC values for all files in rawClips

directory = "C:/Users/thoma/Documents/GitHub/AVCoursework1/rawClips"
files = Path(directory).glob("*")
for filename in files:
  signal, sampleRate = sf.read("rawClips/" + filename.name, dtype='float32')
  name = filename.name.replace(".wav", "")
  featureExtraction.main(signal, sampleRate, name, "MFCCs_07") # Manually iterate per run