import numpy as np
from pathlib import Path
import librosa

def audioMFCC(signal, sampleRate, nfft, win_length, hop_length):
    mfccs = librosa.feature.mfcc(y=signal, sr=sampleRate, n_mfcc=13, n_mels=20, n_fft=nfft, hop_length=hop_length, win_length=win_length)
    return mfccs

def main(signal, sampleRate, filename, folderName):

    # Default values: frameLength 20ms, frameStep 10ms
    frameLength = 0.02
    frameStep = 0.01


    frameLength = int(round(frameLength * sampleRate))
    frameStep = int(round(frameStep * sampleRate))

    # Calculate number of fft
    nfft = 1
    while nfft < frameLength:
        nfft *= 2

    mfcc = audioMFCC(signal, sampleRate, nfft, frameLength, frameStep)

    try:
        Path(folderName).mkdir()
    except FileExistsError:
        print("folder found")
    except PermissionError:
        print("Could not create folder")
    np.save(folderName + "/" + filename + ".npy", mfcc)
