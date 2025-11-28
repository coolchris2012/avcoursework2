from pathlib import Path
import video_feature_extraction
import soundfile as sf

# Generates visual features for all files in Raw_Clips

directory = "C:/Users/thoma/Documents/GitHub/AVCoursework2/Raw_Clips/video"
files = Path(directory).glob("*")
for filename in files:
  name = filename.name.replace(".MOV", "")
  path_to_video = "Raw_Clips/video/" + filename.name
  video_feature_extraction.main(filename, name, "Visual_Features_00") # Manually iterate per run