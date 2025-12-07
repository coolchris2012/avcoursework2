from CNN.Settings import *
from CNN.Dataset.BaseDataset import BaseDataset


class AudioDataset(BaseDataset):
    """
        Dataset for loading pre-computed MFCC features (.npy files) for speech recognition. \n
        Class labels are automatically extracted from filenames by removing digits. \n
        Audio recordings have different lengths, so MFCCs are padded with zeros for batching. \n
        Padding is done in the custom collate function of the DataLoader.
    """
    
    def __init__(self, data_dir: Path, transform=None) -> None:
        super().__init__(data_dir, transform=transform) 
        
        # Load all .npy MFCC files
        self.paths = sorted(list(Path(data_dir).glob("*.npy")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        # Extract unique class names from filenames
        self.classes = self._extract_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.paths)} MFCC files")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def __getitem__(self, index):
        """
        Unique __getitem__ for AudioDataset as it includes adding noise.

        Returns:
            mfcc: Tensor of shape (1, num_frames, 13)
            label_idx: Integer class index
        """
        # Load MFCC from .npy file and preprocess into correct shape
        mfcc_path = self.paths[index]
        mfcc = np.load(mfcc_path).astype(np.float32)  # (13, num_frames) from librosa
        mfcc = mfcc.T  # Transpose to (num_frames, 13)
        
        # Convert to tensor and add channel dimension: (num_frames, 13) -> (1, num_frames, 13)
        mfcc = torch.from_numpy(mfcc).unsqueeze(0)  # (1, num_frames, 13)
        
        # Add noise if enabled in config
        if config.ADD_NOISE:
            mfcc = self._add_noise(mfcc, config.SNR_DB)
        
        # Apply transforms if any (e.g., normalization, augmentation)
        if self.transform:
            mfcc = self.transform(mfcc)
        
        # Get label from filename
        label_name = self._get_label_from_filename(mfcc_path.stem)
        label_idx = self.class_to_idx[label_name]
        
        return mfcc, label_idx
    
    def _add_noise(self, mfcc: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add Gaussian noise to MFCC at specified SNR.
        
        Args:
            mfcc: MFCC tensor of shape (1, num_frames, 13)
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy MFCC tensor
        """
        # Calculate signal power
        signal_power = torch.mean(mfcc ** 2)
        
        # Calculate noise power needed for target SNR
        # SNR_dB = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / (10 ^ (SNR_dB / 10))
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise with calculated power
        noise = torch.randn_like(mfcc) * torch.sqrt(noise_power)
        
        # Add noise to signal
        noisy_mfcc = mfcc + noise
        
        return noisy_mfcc
