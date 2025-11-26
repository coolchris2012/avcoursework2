import numpy as np
from pathlib import Path


def main(signal, sampleRate, filename, folderName):

    # Apply pre-emphasis filter
    preEmphasis = 0.97
    preEmphSig = np.append(signal[0], signal[1:] - preEmphasis * signal[:-1])

    # Default values: frameLength 20ms, frameStep 10ms
    frameLength = 0.02
    frameStep = 0.1


    signalLength = len(preEmphSig)
    frameLength = int(round(frameLength * sampleRate))
    frameStep = int(round(frameStep * sampleRate))
    numFrames = int(np.ceil(float(np.abs(signalLength - frameLength)) / frameStep))

    nFilt = 20 # number of filterbank filters

    # Calculate number of fft
    nfft = 1
    while nfft < frameLength:
        nfft *= 2

    # Pad so number of samples is divisible into equal frames
    padLength = numFrames * frameStep + frameLength
    z = np.zeros((padLength - signalLength))
    padSignal = np.append(preEmphSig, z)

    # Slice into frames
    indices = np.tile(np.arange(0, frameLength), (numFrames, 1)) + np.tile(np.arange(0, numFrames * frameStep, frameStep), (frameLength, 1)).T
    frames = padSignal[indices.astype(np.int32, copy=False)]

    # Apply hamming
    frames = frames * np.hamming(frameLength)

    # Apply fft
    fft = np.fft.rfft(frames, nfft)
    magnitudeFrames = np.abs(fft)
    powerFrames = ((1.0 / nfft) * ((magnitudeFrames) ** 2))

    lowMel = 0
    highMel = 2595 * np.log10(((sampleRate / 2) + 1) / 700)
    melPoints = np.linspace(lowMel, highMel, nFilt + 2)

    # Convert points to bins
    bin = np.floor((nfft + 1) * (700 * (10 ** (melPoints / 2595) - 1)) / sampleRate)

    fBank = np.zeros([nFilt, nfft // 2 + 1])
    for j in range(0, nFilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fBank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fBank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    # Calculate filterbank energies
    feat = np.dot(powerFrames, fBank.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = 20 * np.log10(feat)

    # Apply DCT
    nDct = 13 # Default 13
    mfcc = np.zeros([nFilt, 20])

    for i in range(0, nFilt - 1):
        mfcc[i, :] = feat[i] * (np.cos(nDct * (i + 1 - 0.5) * np.pi / (i + 1)))

    mfcc = mfcc[:, 1 : nDct]

    try:
        Path(folderName).mkdir()
    except FileExistsError:
        print("folder found")
    except PermissionError:
        print("Could not create folder")
    np.save(folderName + "/" + filename + ".npy", mfcc)
