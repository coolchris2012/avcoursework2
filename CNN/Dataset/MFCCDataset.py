from CNN.Settings import *
from CNN.Dataset.BaseDataset import BaseDataset


class MFCCDataset(BaseDataset):
    """
        Dataset for loading pre-computed MFCC features (.npy files) for speech recognition. \n
        Class labels are automatically extracted from filenames by removing digits. \n
        Audio recordings have different lengths, so MFCCs are padded with zeros for batching. \n
        Padding is done in the custom collate function of the DataLoader.
    """
    
    def __init__(self, data_dir: Path, transform=None) -> None:
        super().__init__(data_dir, transform)
        
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
    
    def _extract_classes(self) -> list[str]:
        """Extract unique class names from filenames."""
        class_names = set()
        for path in self.paths:
            # Filename format: "name001.npy" -> extract "name"
            filename = path.stem  # Remove .npy extension
            # Remove trailing digits to get class name
            class_name = ''.join([c for c in filename if not c.isdigit()])
            class_names.add(class_name)
        
        return sorted(list(class_names))
    
    def _get_label_from_filename(self, filename: str) -> str:
        """Extract class label from filename by just removing digits."""
        # Remove extension and digits
        class_name = ''.join([c for c in filename if not c.isdigit()])
        return class_name
    
    def __getitem__(self, index):
        """
        Returns:
            mfcc: Tensor of shape (1, num_frames, 12)
            label_idx: Integer class index
        """
        # Load MFCC from .npy file
        mfcc_path = self.paths[index]
        mfcc = np.load(mfcc_path).astype(np.float32)  # (num_frames, 12)
        
        # Convert to tensor and add channel dimension: (num_frames, 12) -> (1, num_frames, 12)
        mfcc = torch.from_numpy(mfcc).unsqueeze(0)  # (1, num_frames, 12)
        
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
            mfcc: MFCC tensor of shape (1, num_frames, 12)
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
    
    def create_dataloader(self, shuffle: bool, batch_size: int = config.BATCH_SIZE, 
                         num_workers: int = config.NUM_WORKERS):
        """
        Create DataLoader with custom collate function for variable-length MFCCs.
        """
        return DataLoader(
            self, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle,
            collate_fn=self.collate_fn  # Use custom collate for padding
        )
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length MFCC sequences.
        Pads all sequences to the maximum length in the batch.
        This ensures that all input tensors have the same shape.
        
        Args:
            batch: List of tuples (mfcc, label_idx)
        
        Returns:
            mfccs: Padded tensor of shape (batch_size, 1, max_frames, 12)
            labels: Tensor of shape (batch_size,)
        """
        mfccs, labels = zip(*batch)
        
        # Find maximum number of frames in this batch
        max_frames = max(mfcc.shape[1] for mfcc in mfccs)
        
        # Pad all MFCCs to max_frames
        padded_mfccs = []
        for mfcc in mfccs:
            # mfcc shape: (1, num_frames, 12)
            current_frames = mfcc.shape[1]
            pad_amount = max_frames - current_frames
            
            # Pad along the frame dimension (dim=1)
            # Padding format for pad(): (last_dim_left, last_dim_right, second_last_left, second_last_right, ...)
            # Shape is (1, num_frames, 12), so to pad num_frames: (0, 0, 0, pad_amount)
            if pad_amount > 0:
                padded = torch.nn.functional.pad(mfcc, (0, 0, 0, pad_amount), value=0)
            else:
                padded = mfcc
            
            padded_mfccs.append(padded)
        
        # Stack into batch tensor
        mfccs_batch = torch.stack(padded_mfccs, dim=0)  # (batch_size, 1, max_frames, 12)
        labels_batch = torch.LongTensor(labels)  # (batch_size,)
        
        return mfccs_batch, labels_batch
