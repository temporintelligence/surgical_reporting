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


def decode_vector(vectors, json_annotations_path, surgery_mappings):
    """
    Convert JSON vectors into a binary vector of size 21 indicating instrument/target presence.
    
    Args:
        vectors: List of vectors to decode
        json_annotations_path: Path to JSON annotations file
        surgery_mappings: SurgeryMappings object containing reverse_mapping
        
    Returns:
        numpy.ndarray: Binary vector of size 21
    """
    # Initialize result vector with zeros
    result = np.zeros(21, dtype=np.int8)
    
    # Load JSON data only once (outside the loop)
    try:
        with open(json_annotations_path, 'r') as file:
            data = json.load(file)
        triplets = data['categories']['triplet']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Error loading JSON annotations: {str(e)}")
    
    for vector in vectors:
        idx = int(vector[0])
        if idx == -1:
            return result  # Early exit for non-existing value
            
        try:
            triplet = triplets[str(idx)]
            instrument, _, target = triplet.split(',')
            
            # Get instrument and target numbers
            instrument_number = surgery_mappings.reverse_mapping.get(instrument, -1)
            target_number = surgery_mappings.reverse_mapping.get(target, -1)
            
            # Update result vector
            if 0 <= instrument_number < 21:
                result[instrument_number] = 1
            if 0 <= target_number < 21:
                result[target_number] = 1
                
        except (KeyError, ValueError, AttributeError) as e:
            raise ValueError(f"Error processing vector {vector}: {str(e)}")
    
    return result

def generate_action_description(annotations, json_annotations_path):
    """
    Generates a natural language description of surgical actions from annotation data.
    
    Args:
        annotations: List of annotation vectors where each vector contains: [triplet_index, ..., phase_index]
        json_annotations_path: Path to JSON file containing surgical action definitions
    
    Returns:
        str: Formatted sentence describing the actions and phase, or "Unknown" if invalid
    """
    # Initialize tracking variables
    num_annotations = len(annotations)
    current_annotation = 1  # 1-based counter
    
    # Process each annotation vector
    for vector in annotations:
        # Extract triplet and phase indices
        triplet_index = vector[0]
        phase_index = vector[-1]
        
        # Handle unknown action case
        if triplet_index == -1:
            return "Unknown"
        
        # Load and parse JSON data
        with open(json_annotations_path, 'r') as file:
            data = json.load(file)
        
        # Get action components
        triplet = data['categories']['triplet'][str(triplet_index)]
        instrument, verb, target = triplet.split(',')
        
        # Get phase information
        phase = data['categories']['phase'][str(phase_index)]
        
        # Print current action description
        print(f"The {instrument} is {verb}ing the {target}", end="")
        
        # Handle sentence connectors
        if current_annotation < num_annotations:
            print(", and ", end="")  # Connect multiple actions
        elif current_annotation == num_annotations:
            print(f" during phase {phase}.", end="")  # Final phrase
        
        current_annotation += 1
    
    print()  # Final newline

