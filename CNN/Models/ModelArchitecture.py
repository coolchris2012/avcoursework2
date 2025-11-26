from CNN.Settings import *


class SpeechClassifier(nn.Module):
    """
    CNN model for classifying speech from MFCC features.
    Takes variable-length MFCC sequences and outputs class probabilities.
    
    LIMITATION: This architecture assumes all inputs are valid vocabulary words (one of the 20 names).
    Out-of-vocabulary inputs (e.g., different words, noise, silence) will still be forced into
    one of the 20 classes, potentially causing misclassification.
    
    A solution would be to add an objectness/confidence mechanism similar to YOLO:
        1. Add a dual-output head that predicts both objectness score and class logits
        2. Collect training data with "background" or out-of-vocabulary examples  
        3. Use a combined loss function: BCE(objectness) + CrossEntropy(classes)
    
    For isolated word recognition in controlled demo conditions where all inputs are from the
    known vocabulary, this simpler classification-only approach is appropriate.
    """

    # ---------------Template for each convolutional block that will be used------------------
    def conv_block(self, in_ch, out_ch, do_pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
        ]
        if do_pool:
            layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def __init__(self,
                 num_classes: int = 20,
                 hidden_units: int = config.HIDDEN_UNITS,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_units = hidden_units

    
        #-----------------------------Full model architecture-----------------------------
        
        # Feature extraction blocks - dynamically create based on NUM_LAYERS
        for i in range(config.NUM_LAYERS):
            if i == 0:
                in_channels = 1  # First block takes MFCC input
            else:
                in_channels = hidden_units * (2 ** (i - 1))
            
            out_channels = hidden_units * (2 ** i)
            
            # Only pool in first 2 blocks to prevent spatial collapse
            # With input (20, 12): pool twice -> (5, 3) which is safe
            do_pool = (i < 2)
            
            self.add_module(f'block{i+1}', self.conv_block(in_channels, out_channels, do_pool=do_pool))
        
        # Calculate final number of channels
        final_channels = hidden_units * (2 ** (config.NUM_LAYERS - 1))
        
        # Global pooling to handle variable frame lengths
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_channels // 2, num_classes)
        )
        
        #----------------------------------------------------------------------------------

    def forward(self, x):
        # x: (B, 1, num_frames, 12)
        
        # 1) Run through feature extraction blocks
        for i in range(config.NUM_LAYERS):
            x = getattr(self, f'block{i+1}')(x)
        
        # 2) Global pooling to handle variable lengths
        x = self.global_pool(x)
        
        # 3) Classification head
        x = self.classifier(x)
        
        return x  # Raw logits for CrossEntropyLoss
