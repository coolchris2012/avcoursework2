from CNN.Settings.CNNImports import *

# Model settings
HIDDEN_UNITS = 32
NUM_CLASSES = 20  # Number of names to recognize
NUM_LAYERS = 3  # Number of convolutional layers

# Training settings
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 50
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Optimizer settings
OPTIMIZER = torch.optim.AdamW  # Optimizer class
OPTIMIZER_PARAMS: dict = {  # type: ignore
    'lr': 1e-3,  # Learning rate
    'weight_decay': 1e-4  # L2 regularization
}

# Loss function
LOSS_FUNCTION = nn.CrossEntropyLoss

# Learning rate scheduler settings
LR_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau
LR_SCHEDULER_PARAMS: dict = {  # type: ignore
    'mode': 'min',  # Monitor training loss (minimize)
    'factor': 0.5,  # Reduce LR by half
    'patience': 5   # Wait 5 epochs before reducing
}

# Paths
MFCC_DIR = Path("MFCCs_06")  # Directory containing MFCC .npy files

# Noise settings
ADD_NOISE = False  
SNR_DB = 10  # Signal-to-noise ratio in dB (used when ADD_NOISE=True)

# Model saving - Create timestamped folder for each run
from datetime import datetime
RUN_TIMESTAMP = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
RUN_FOLDER = Path("CNN/Models") / RUN_TIMESTAMP
MODEL_SAVE_PATH = RUN_FOLDER / "best_model.pth"

# Transformations (currently None for MFCCs, could add augmentation later)
TRAIN_TRANSFORMATION = None
TEST_TRANSFORMATION = None

