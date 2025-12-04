# Torch
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Torchvision
import torchvision
from torchvision import transforms

#Visualization libraries
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# General
from enum import Enum, auto
import sys
import time

#File handling libraries
import os
import pathlib
from pathlib import Path
import pickle
import gzip
import argparse

# Data manipulation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Maths
import numpy as np
import pandas as pd
import random
import math



# Dummy MFCC data for testing
dummy_mfccs = torch.randn(100, 40, 50, 1)  # [batch, coeffs, time, channel]
dummy_labels = torch.randint(0, 20, (100,))  # 20 classes