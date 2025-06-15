import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from transformers import DetrForObjectDetection, DetrImageProcessor


# Saving and model directory
save_dir = "data/save"
model_dir = "weights/"

# Load the test dataset
test_dataset = torch.load(
    f"{save_dir}/datasets/frame_dataset_0_0.pt", weights_only=False
)


def collate_fn(batch):
    """
    Custom collate function for DataLoader that batches frame-based data.

    Args:
        batch (list of dict): Each dict represents one data sample with keys:
            - 'frame' (Tensor): Image tensor of shape [3, H, W].
            - 'frame_caption' (str): Caption describing the frame.
            - 'object_labels' (list of int): Multi-label binary classification vector.
            - 'objects' (list of str): List of object names present in the frame.

    Returns:
        dict: A batch dictionary with the following keys:
            - 'frame' (Tensor): Stacked image tensors of shape [batch_size, 3, H, W].
            - 'frame_caption' (list of str): List of frame-level captions.
            - 'object_labels' (Tensor): Stacked object label tensors of shape [batch_size, num_labels].
            - 'objects' (list of list of str): List of object names for each sample,
              padded or truncated to `max_objects` (default: 10).
    """
    max_objects = 10

    return {
        "frame": torch.stack([item["frame"] for item in batch]),
        "frame_caption": [item["frame_caption"] for item in batch],
        "object_labels": torch.stack(
            [torch.tensor(item["object_labels"]) for item in batch]
        ),
        "objects": [
            (
                item["objects"] + [""] * (max_objects - len(item["objects"]))
                if len(item["objects"]) < max_objects
                else item["objects"][:max_objects]
            )
            for item in batch
        ],
    }


# Create the test dataloader
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
