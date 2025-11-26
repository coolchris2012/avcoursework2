from CNN.Settings import *
from CNN.Dataset.BaseDataset import BaseDataset


class AudioVisualDataset(BaseDataset):
    """
        Dataset for loading pre-computed merged features (.npy files) for lip-reading. \n
        Class labels are automatically extracted from filenames by removing digits. \n
        Videos have different lengths, so features are padded with zeros for batching. \n
        Padding is done in the custom collate function of the DataLoader.
    """
    
    def __init__(self, data_dir: Path, transform=None) -> None:
        super().__init__(data_dir, transform)
        
        # Load all .npy visual feature files
        self.paths = sorted(list(Path(data_dir).glob("*.npy")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        # Extract unique class names from filenames
        self.classes = self._extract_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.paths)} AudioVisual feature files")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def __getitem__(self, index):
        """
        Unique __getitem__ for AudioVisualDataset as it does NOT include noise addition.

        Returns:
            audiovisual: Tensor of shape (1, num_frames, num_features)
            label_idx: Integer class index
        """
        # Load merged audio+visual features from .npy file
        audiovisual_path = self.paths[index]
        audiovisual = np.load(audiovisual_path).astype(np.float32)  # (num_frames, num_features)
        
        # Convert to tensor and add channel dimension: (num_frames, N) -> (1, num_frames, N)
        audiovisual = torch.from_numpy(audiovisual).unsqueeze(0)
        
        # Apply transforms if any (e.g., normalization, augmentation)
        if self.transform:
            audiovisual = self.transform(audiovisual)
        
        # Get label from filename
        label_name = self._get_label_from_filename(audiovisual_path.stem)
        label_idx = self.class_to_idx[label_name]
        
        return audiovisual, label_idx
