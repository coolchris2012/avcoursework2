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
    



    def create_dataloader(self,
                    shuffle: bool,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    ):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


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
    