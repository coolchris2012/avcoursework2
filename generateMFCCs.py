from pathlib import Path
import featureExtraction
import soundfile as sf

# Generates MFCC values for all files in Raw_Clips

directory = "C:/Users/thoma/Documents/GitHub/AVCoursework2/Raw_Clips/audio"
files = Path(directory).glob("*")
for filename in files:
  signal, sampleRate = sf.read("Raw_Clips/audio/" + filename.name, dtype='float32')
  name = filename.name.replace(".wav", "")
  featureExtraction.main(signal, sampleRate, name, "MFCCs_00") # Manually iterate per run