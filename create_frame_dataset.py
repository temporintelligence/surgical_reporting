import json
import os
import re

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# Object dictionary
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

# Reverses the object dictionary in order to get the key
reverse_mapping = {v: k for k, v in mapping.items()}


# List of important data as well as saving directories
raw_data_dir = "data/CholecT50"
raw_video_dir = raw_data_dir + "/videos"
raw_annotations_dir = raw_data_dir + "/labels"
save_dir = "data/save"


def get_objects(frame_annotation):
    """
    Extracts and returns a list of non-null instruments and targets
    from a frame's annotation vectors.

    Args:
        frame_annotation (list of list[int]):
            A list of annotation vectors for a given video frame.
            The first element of each vector contains the the index of a triplet
            that refers to a structured (instrument, verb, target) annotation.

    Returns:
        list[str] or str:
            A list of strings representing the instruments and targets
            (excluding any "null_instrument" or "null_target" labels) found in the triplets.
            Returns the string "Unknown" if a triplet index is -1.
    """
    objects = []
    for annotation_vector in frame_annotation:
        triplet_idx = annotation_vector[0]

        if triplet_idx == -1:
            return "Unknown"

        # Note: it does not matter which annotation file we choose here,
        # the triplets indexing is always the same
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
    """
    Generates a natural language caption describing the actions in a frame
    based on the annotated triplets and surgical phase.

    Args:
        frame_annotation (list of list[int]):
            Each annotation vector contains a triplet index and a phase index.
            The triplet index refers to an (instrument, verb, target) combination.
            The phase index refers to the current surgical phase.

    Returns:
        str:
            A descriptive sentence summarizing the actions in the frame.
            Returns "Unknown" if any triplet index is -1.
    """
    n = len(frame_annotation)
    i = 1
    result = ""

    for annotation_vector in frame_annotation:
        triplet_idx = annotation_vector[0]
        phase_idx = annotation_vector[-1]

        if triplet_idx == -1:
            return "Unknown"

        # Note: it does not matter which annotation file we choose here,
        # the triplets indexing is always the same
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
    """
    Loads and preprocesses a video frame for model input.

    Args:
        frame_path (str):
            Path to the video frame file.
        target_size (tuple of int):
            Desired (width, height) for resizing the image. Default is (224, 224).

    Returns:
        torch.Tensor:
            A tensor representation of the frame, normalized to [0,1] with shape (3, H, W).
    """
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
    """
    Converts a frame's annotation into a multi-hot label vector.

    Args:
        frame_annotation (list of list[int]):
            Each annotation vector includes a triplet index referring to
            an (instrument, verb, target) triplet.

    Returns:
        np.ndarray:
            A binary vector of shape (21,), where each index indicates the
            presence (1) or absence (0) of a specific instrument or target.
    """
    labels = np.zeros(21)

    # Note: it does not matter which annotation file we choose here,
    # the triplets indexing is always the same
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


def create_frame_dataset(
    raw_video_dir, raw_annotations_dir, start_video=None, end_video=None
):
    """
    Creates a dataset of annotated and preprocessed frames from videos.

    Args:
        raw_video_dir (str):
            Path to the directory containing folders of extracted video frames.
        raw_annotations_dir (str):
            Path to the directory containing JSON annotation files (one per video).
        start_video (int, optional):
            Index of the first video to include (for slicing the video folder list).
        end_video (int, optional):
            Index of the last video to include (exclusive).

    Returns:
        list of dict:
            Each dict contains:
                - "video": ID of the video
                - "frame_number": frame filename (without extension)
                - "frame": preprocessed image tensor
                - "object_labels": multi-hot vector of object/target presence
                - "objects": list of detected instrument/target names
                - "frame_caption": generated natural language description
    """
    dataset = []

    all_video_folders = sorted(os.listdir(raw_video_dir))
    if start_video is not None or end_video is not None:
        all_video_folders = all_video_folders[start_video:end_video]

    for video_folder in all_video_folders:
        print(f"Processing: {video_folder}")
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
    start_video = 10
    end_video = 20

    dataset = create_frame_dataset(
        raw_video_dir, raw_annotations_dir, start_video=start_video, end_video=end_video
    )

    torch.save(
        dataset, f"{save_dir}/datasets/frame_dataset_{start_video}_{end_video-1}.pt"
    )


if __name__ == "__main__":
    main()
