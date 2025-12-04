from CNN.Settings import *
from CNN.Dataset.BaseDataset import BaseDataset


class AudioVisualDataset(BaseDataset):
    """
    Dataset for audio-visual early fusion.
    Loads audio from MFCCs_00/ and visual from Visual_Features_00/.
    Processes and merges them on-the-fly during training.
    """
    
    def __init__(self, audio_dir: Path, visual_dir: Path, transform=None) -> None:
        """
        Args:
            audio_dir: Path to audio features (MFCCs_00/)
            visual_dir: Path to visual features (Visual_Features_00/)
            transform: Optional transform to apply
        """
        # Call super with both directories
        super().__init__(Path(audio_dir), Path(visual_dir), transform=transform)
        
        # Store references for convenience
        self.audio_dir = self.data_dirs[0]
        self.visual_dir = self.data_dirs[1]
        
        # Load all .npy audio files (use self.paths for BaseDataset compatibility)
        self.paths = sorted(list(self.audio_dir.glob("*.npy")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {audio_dir}")
        
        # Verify matching visual files exist
        missing = []
        for audio_path in self.paths:
            visual_path = self.visual_dir / audio_path.name
            if not visual_path.exists():
                missing.append(audio_path.name)
        
        if missing:
            raise ValueError(f"Missing visual files for: {missing[:5]}...")
        
        # Extract unique class names from filenames
        self.classes = self._extract_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.paths)} audio-visual pairs")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    

    # Class extraction methods
    def _extract_classes(self):
        """Extract all the unique class names using a set from filenames by removing digits."""
        import re
        classes = set()
        for path in self.paths:
            # Remove digits to get class name
            class_name = re.sub(r'\d+', '', path.stem)
            classes.add(class_name)
        return sorted(list(classes))
    
    def _get_label_from_filename(self, filename):
        """Get one label per sample by removing digits from filename."""
        import re
        return re.sub(r'\d+', '', filename)
    


    
    def _average_pool_2d(self, arr, pool_size=2):
        """Apply 2x2 average pooling to 3D array (frames, height, width)"""
        num_frames, h, w = arr.shape
        
        # Crop to make dimensions divisible by pool_size
        new_h = (h // pool_size) * pool_size
        new_w = (w // pool_size) * pool_size
        arr_cropped = arr[:, :new_h, :new_w]
        
        # Calculate output dimensions
        out_h, out_w = new_h // pool_size, new_w // pool_size
        
        # Reshape to create pooling blocks
        reshaped = arr_cropped.reshape(num_frames, out_h, pool_size, out_w, pool_size)
        # Average over the pooling dimensions
        pooled = reshaped.mean(axis=(2, 4))
        
        return pooled
    
    def _process_visual(self, visual, target_frames):
        """Process visual: transpose, pool, flatten, upsample to audio rate
        Input: (60, 66, 200)
        Output: (target_frames, 990)
        """
        # Transpose to (frames, height, width)
        visual = visual.transpose(2, 0, 1)  # (200, 60, 66)
        
        # Apply 2x2 average pooling
        visual = self._average_pool_2d(visual, pool_size=2)  # (200, 30, 33)
        
        # Flatten spatial dimensions
        num_frames = visual.shape[0]
        visual = visual.reshape(num_frames, -1)  # (200, 990)
        
        # Upsample to match audio frame rate by repeating frames
        repeat_factor = target_frames // num_frames
        remainder = target_frames % num_frames
        
        # Repeat each frame
        visual_upsampled = np.repeat(visual, repeat_factor, axis=0)
        
        # Handle remainder by repeating first frames once more
        if remainder > 0:
            visual_upsampled = np.concatenate([visual_upsampled, visual[:remainder]], axis=0)
        
        return visual_upsampled
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        """
        Load audio and visual, process and merge on-the-fly.

        Returns:
            merged: Tensor of shape (1, num_frames, 1003) where num_frames varies
            label_idx: Integer class index
        """
        # Load audio features
        audio_path = self.paths[index]
        audio = np.load(audio_path).astype(np.float32)  # (N, 13, 1)
        
        # Remove channel dimension
        audio = np.squeeze(audio, axis=-1) if audio.ndim == 3 else audio  # (N, 13)
        
        # Load visual features
        visual_path = self.visual_dir / audio_path.name
        visual = np.load(visual_path).astype(np.float32)  # (60, 66, 200)
        
        # Process visual and upsample to match audio frame count
        target_frames = audio.shape[0]
        visual_processed = self._process_visual(visual, target_frames)  # (N, 990)
        
        # Concatenate along feature dimension
        merged = np.concatenate([audio, visual_processed], axis=1)  # (N, 1003)
        
        # Convert to tensor and add channel dimension: (N, 1003) -> (1, N, 1003)
        merged = torch.from_numpy(merged).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            merged = self.transform(merged)
        
        # Get label from filename
        label_name = self._get_label_from_filename(audio_path.stem)
        label_idx = self.class_to_idx[label_name]
        
        return merged, label_idx
