import json
import os
import pickle
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


instrument_mapping = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "specimen_bag",
    7: "no_instrument",
}

verb_mapping = {
    0: "grasp",
    1: "retract",
    2: "dissect",
    3: "coagulate",
    4: "clip",
    5: "cut",
    6: "aspirate",
    7: "irrigate",
    8: "pack",
    9: "null_verb",
}

target_mapping = {
    0: "gallbladder",
    1: "cystic_plate",
    2: "cystic_duct",
    3: "cystic_artery",
    4: "cystic_pedicle",
    5: "blood_vessel",
    6: "fluid",
    7: "abdominal_wall_cavity",
    8: "liver",
    9: "adhesion",
    10: "omentum",
    11: "peritoneum",
    12: "gut",
    13: "specimen_bag",
    14: "null_target",
}

phase_mapping = {
    0: "preparation",
    1: "calot-triangle-dissection",
    2: "clipping-and-cutting",
    3: "gallbladder-dissection",
    4: "gallbladder-packaging",
    5: "cleaning-and-coagulation",
    6: "gallbladder-extraction",
}

mapping = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "specimen_bag",
    7: "gallbladder",
    8: "cystic_plate",
    9: "cystic_duct",
    10: "cystic_artery",
    11: "cystic_pedicle",
    12: "blood_vessel",
    13: "fluid",
    14: "abdominal_wall_cavity",
    15: "liver",
    16: "adhesion",
    17: "omentum",
    18: "peritoneum",
    19: "gut",
    20: "null_target",
}

reverse_mapping = {v: k for k, v in mapping.items()}


raw_data_dir = "data/CholecT50"
raw_video_dir = raw_data_dir + "/videos"
raw_annotations_dir = raw_data_dir + "/labels"

save_dir = "data/save"


def get_objects(frame_annotation):
    objects = []
    for annotation_vector in frame_annotation:
        triplet_idx = annotation_vector[0]

        if triplet_idx == -1:
            return "Unknown"

        # Note: it does not matter which file we choose here, the triplets indexing is always the same
        with open(raw_annotations_dir + "/VID01.json", "r") as file:
            data = json.load(file)
        triplets = data["categories"]["triplet"]
        triplet = triplets[str(triplet_idx)]
        instrument, _, target = triplet.split(",")

        if instrument != "null_instrument":
            objects.append(str(instrument))
        if target != "null_target":
            objects.append(str(target))

    return objects


def get_frame_caption(frame_annotation):
    n = len(frame_annotation)
    i = 1
    result = ""

    for annotation_vector in frame_annotation:
        triplet_idx = annotation_vector[0]
        phase_idx = annotation_vector[-1]

        if triplet_idx == -1:
            return "Unknown"

        with open(raw_annotations_dir + "/VID01.json", "r") as file:
            data = json.load(file)

        triplets = data["categories"]["triplet"]
        triplet = triplets[str(triplet_idx)]
        instrument, verb, target = triplet.split(",")

        phases = data["categories"]["phase"]
        phase = phases[str(phase_idx)]

        if i == 1:
            result += f" During phase {phase}, "

        if verb != "null_verb":
            if verb[-1] == "e":
                verb = verb[:-1]
            if instrument == "scissors":
                result += f"the {instrument} are {verb}ing the {target}"
            else:
                result += f"the {instrument} is {verb}ing the {target}"
        else:
            result += f"the {instrument} is present"

        if i < n:
            result += ", "
        i += 1

    return result


def preprocess_frame(frame_path, target_size=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(frame_path).convert("RGB")
    tensor = transform(image)

    return tensor


def annotation_to_label(frame_annotation):
    labels = np.zeros(21)

    with open(raw_annotations_dir + "/VID01.json", "r") as file:
        data = json.load(file)

    for annotation_vector in frame_annotation:
        triplet_idx = int(annotation_vector[0])

        if triplet_idx == -1:
            return labels

        triplets = data["categories"]["triplet"]
        triplet = triplets[str(triplet_idx)]
        instrument, _, target = triplet.split(",")

        instrument_number = reverse_mapping.get(instrument)
        target_number = reverse_mapping.get(target)

        if instrument_number != 20 and target_number != 20:
            labels[instrument_number] = 1
            labels[target_number] = 1

    return labels


def create_dataset(
    raw_video_dir, raw_annotations_dir, start_video=None, end_video=None
):
    dataset = []

    all_video_folders = sorted(os.listdir(raw_video_dir))
    if start_video is not None or end_video is not None:
        all_video_folders = all_video_folders[start_video:end_video]

    for video_folder in all_video_folders:
        video_path = os.path.join(raw_video_dir, video_folder)

        if not os.path.isdir(video_path):
            continue

        frames = sorted(os.listdir(video_path))
        frames = [f for f in frames if f.endswith((".png", ".jpg", ".jpeg"))]
        processed_frames = set()

        idx = 0
        while idx < len(frames):
            frame_name = frames[idx]

            if frame_name in processed_frames:
                idx += 1
                continue

            processed_frames.add(frame_name)
            match = re.match(r"^(\d+)", os.path.splitext(frame_name)[0])

            if not match:
                print(f"Skipping invalid frame name: {frame_name}")
                idx += 1
                continue

            frame_number = int(match.group(1))

            frame_path = os.path.join(video_path, frame_name)
            video_id = os.path.basename(os.path.dirname(frame_path))
            video_annotation = os.path.join(raw_annotations_dir, f"{video_id}.json")

            with open(video_annotation, "r") as f:
                frame_annotations = json.load(f)

            frame_key = str(frame_number)
            frame_annotation = frame_annotations["annotations"].get(frame_key, None)
            frame_name = frame_name.replace(".png", "")

            objects = get_objects(frame_annotation)
            frame_caption = get_frame_caption(frame_annotation)
            frame = preprocess_frame(frame_path)
            object_labels = annotation_to_label(frame_annotation)

            if frame_caption != "Unknown":
                dataset.append(
                    {
                        "video": video_folder,
                        "frame_number": frame_name,
                        "frame": frame,
                        "object_labels": object_labels,
                        "objects": objects,
                        "frame_caption": frame_caption,
                    }
                )

            idx += 1

    return dataset


def main():
    start_video = 0
    end_video = 1

    dataset = create_dataset(
        raw_video_dir, raw_annotations_dir, start_video=start_video, end_video=end_video
    )

    torch.save(
        dataset, f"{save_dir}/datasets/frame_dataset_{start_video}_{end_video-1}.pt"
    )


if __name__ == "__main__":
    main()
