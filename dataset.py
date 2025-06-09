import os
import re
import sys
import json
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F

import torch


current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data")

cholec_dir = os.path.join(data_dir, "CholecT50")
save_dir = os.path.join(data_dir, "save")

video_dir = os.path.join(cholec_dir, "videos")
json_annotations = os.path.join(cholec_dir, "labels", "VID01.json")
labels_dir = os.path.join(cholec_dir, "labels")

os.makedirs(save_dir, exist_ok=True)

