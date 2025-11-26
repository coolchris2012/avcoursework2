from CNN.Settings import *

class BaseDataset(Dataset):
    """
    Base dataset class for PyTorch datasets.
    
    Provides basic functionality for loading data, creating dataloaders,
    and splitting datasets into training and testing sets.
    
    Attributes:
        data_dir: Path to the directory containing the dataset
        transform: Optional transform to be applied to samples
        paths: List of file paths to dataset samples
    """
    
    def __init__(self, data_dir: Path, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.paths: list[Path] = []  # each dataset will fill

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        raise NotImplementedError("Child classes must implement __getitem__()")
    
    def _extract_classes(self) -> list[str]:
        """Extract unique class names from filenames by removing digits."""
        class_names = set()
        for path in self.paths:
            # Filename format: "name001.npy" -> extract "name"
            filename = path.stem  # Remove .npy extension
            # Remove trailing digits to get class name
            class_name = ''.join([c for c in filename if not c.isdigit()])
            class_names.add(class_name)
        
        return sorted(list(class_names))
    
    def _get_label_from_filename(self, filename: str) -> str:
        """Extract class label from filename by removing digits."""
        class_name = ''.join([c for c in filename if not c.isdigit()])
        return class_name
    



    def create_dataloader(self,
                    shuffle: bool,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    ):
        return DataLoader(
            self, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle,
            collate_fn=self.collate_fn  # Use custom collate for variable-length padding
        )
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences.
        Pads all sequences to the maximum length in the batch.
        
        Args:
            batch: List of tuples (features, label_idx)
        
        Returns:
            features_batch: Padded tensor of shape (batch_size, 1, max_frames, num_features)
            labels_batch: Tensor of shape (batch_size,)
        """
        features, labels = zip(*batch)
        
        # Find maximum number of frames in this batch
        max_frames = max(feat.shape[1] for feat in features)
        
        # Pad all features to max_frames
        padded_features = []
        for feat in features:
            # feat shape: (1, num_frames, num_features)
            current_frames = feat.shape[1]
            pad_amount = max_frames - current_frames
            
            # Pad along the frame dimension (dim=1)
            if pad_amount > 0:
                padded = torch.nn.functional.pad(feat, (0, 0, 0, pad_amount), value=0)
            else:
                padded = feat
            
            padded_features.append(padded)
        
        # Stack into batch tensor
        features_batch = torch.stack(padded_features, dim=0)
        labels_batch = torch.LongTensor(labels)
        
        return features_batch, labels_batch


    @classmethod
    def from_split(
                    cls,
                    data_dir: Path,
                    *args, # <-------------------- Placeholder for any additional arguments
                    train_transform=config.TRAIN_TRANSFORMATION,
                    test_transform=config.TEST_TRANSFORMATION,
                    split_ratio: float = config.TRAIN_SPLIT_RATIO,
                    seed: int = config.RANDOM_SEED,
                    **kwargs # <-------------------- Placeholder for any key worded arguments
                  ):

        # 1) Create a temporary "full" dataset
        full_dataset = cls(data_dir, *args, **kwargs)
        all_paths = full_dataset.paths 

        if len(all_paths) == 0:
            raise ValueError("No paths found in dataset, cannot split.")

        # 2) Split into train/test
        train_paths, test_paths = train_test_split(
            all_paths, 
            train_size=split_ratio, 
            shuffle=True, 
            random_state=seed
        )

        # 3) Create train dataset
        train_ds = cls(data_dir, *args, transform=train_transform, **kwargs)
        train_ds.paths = train_paths  # override with the split

        # 4) Create test dataset
        test_ds  = cls(data_dir, *args, transform=test_transform, **kwargs)
        test_ds.paths = test_paths

        return train_ds, test_ds
    