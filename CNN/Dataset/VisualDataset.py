from CNN.Settings import *
from CNN.Dataset.BaseDataset import BaseDataset


class VisualDataset(BaseDataset):
    """
        Dataset for loading pre-computed visual features (.npy files) for lip-reading. \n
        Class labels are automatically extracted from filenames by removing digits. \n
        Videos have different lengths, so features are padded with zeros for batching. \n
        Padding is done in the custom collate function of the DataLoader.
    """
    
    def __init__(self, data_dir: Path, transform=None) -> None:
        super().__init__(data_dir, transform=transform)
        
        # Load all .npy visual feature files
        self.paths = sorted(list(Path(data_dir).glob("*.npy")))
        
        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        # Extract unique class names from filenames
        self.classes = self._extract_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.paths)} visual feature files")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def __getitem__(self, index):
        """
        Unique __getitem__ for VisualDataset as it does NOT include noise addition.

        Returns:
            visual: Tensor of shape (1, num_frames, num_features)
            label_idx: Integer class index
        """
        # Load visual features from .npy file and preprocess into correct shape
        visual_path = self.paths[index]
        visual = np.load(visual_path).astype(np.float32)  # (height=60, width=66, frames=200)
        visual = visual.transpose(2, 0, 1)  # Transpose: (height, width, frames) -> (frames, height, width)
        visual = self._average_pool_2d(visual, pool_size=2) # Apply 2x2 average pooling (200, 60, 66) -> (200, 30, 33)
        
        num_frames = visual.shape[0] # Flatten spatial dimensions: (200, 30, 33) -> (200, 990)
        visual = visual.reshape(num_frames, -1)
        visual = torch.from_numpy(visual).unsqueeze(0) # Convert to tensor AND add channel dimension: (200, 990) -> (1, 200, 990)
        


        # Apply transforms if any (e.g., normalization, augmentation)
        if self.transform:
            visual = self.transform(visual)
        
        # Get label from filename
        label_name = self._get_label_from_filename(visual_path.stem)
        label_idx = self.class_to_idx[label_name]
        
        return visual, label_idx
    
    def _average_pool_2d(self, data, pool_size=2):
        """
        Apply 2D average pooling to reduce spatial dimensions like how CNNs do it.
        
        Args:
            data: numpy array of shape (frames, height, width)
            pool_size: size of pooling window (default: 2 for 2x2 pooling)
        
        Returns:
            Pooled array of shape (frames, height//pool_size, width//pool_size)
        """
        num_frames, height, width = data.shape
        new_height = height // pool_size
        new_width = width // pool_size
        
        # Reshape and average
        pooled = data[:, :new_height*pool_size, :new_width*pool_size]
        pooled = pooled.reshape(num_frames, new_height, pool_size, new_width, pool_size)
        pooled = pooled.mean(axis=(2, 4))  # Average over pool_size dimensions
        
        return pooled
