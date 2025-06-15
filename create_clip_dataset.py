import json
import os
import re

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


raw_data_dir = "data/CholecT50"
raw_video_dir = raw_data_dir + "/videos"
raw_annotations_dir = raw_data_dir + "/labels"

save_dir = "data/save"


def load_and_preprocess_frames(clip, target_size=(224, 224), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )

    frames = []

    for frame in clip:
        image = Image.open(frame).convert("RGB")
        tensor = transform(image).to(device)
        frames.append(tensor)

    clip_tensor = torch.stack(frames).to(device)
    return clip_tensor


def get_frame_captions(frame_paths):
    captions = []

    for frame_path in frame_paths:
        video_id = os.path.basename(os.path.dirname(frame_path))
        frame_number = os.path.splitext(os.path.basename(frame_path))
        frame_number = int(frame_number[0][:6])

        annotation_file = os.path.join(raw_annotations_dir, f"{video_id}.json")

        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        frame_idx = str(frame_number)
        frame_annotation = annotations["annotations"].get(frame_idx, None)

        caption = get_frame_caption(frame_annotation)
        captions.append(caption)

    return captions


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


def extract_phase_and_actions(captions):
    results = []
    prev_phase = None
    grouped_actions = []
    time_count = 0

    for caption in captions:
        phase_match = re.search(r"during phase ([a-zA-Z\-]+)", caption, re.IGNORECASE)
        phase = phase_match.group(1) if phase_match else prev_phase

        if phase is None:
            continue

        actions_part = re.sub(
            r"during phase [a-zA-Z\-]+,? ", "", caption, flags=re.IGNORECASE
        )
        actions = [
            action.strip()
            for action in actions_part.split(",")
            if action.strip() and action.lower() != "unknown"
        ]

        if phase == prev_phase:
            grouped_actions = list(set(grouped_actions + actions))
            time_count += 1
            results[-1]["actions"] = grouped_actions
            results[-1]["time"] = time_count
        else:
            grouped_actions = actions
            time_count = 1
            results.append(
                {"phase": phase, "actions": grouped_actions, "time": time_count}
            )

        prev_phase = phase

    return create_sentence(results)


def create_clip_dataset(raw_video_dir, clip_length, overlap, start_video, end_video):
    dataset = []

    video_folders = sorted(
        [
            video_folder
            for video_folder in os.listdir(raw_video_dir)
            if os.path.isdir(os.path.join(raw_video_dir, video_folder))
        ]
    )

    selected_videos = video_folders[start_video:end_video]

    for video_folder in selected_videos:
        print(f"Processing: {video_folder}")
        video_path = os.path.join(raw_video_dir, video_folder)

        frames = sorted(
            [
                frame
                for frame in os.listdir(video_path)
                if frame.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        start_frame_idx = 0

        while start_frame_idx + clip_length <= len(frames):
            frame_paths = [
                os.path.join(video_path, frames[frame_idx])
                for frame_idx in range(start_frame_idx, start_frame_idx + clip_length)
            ]

            frame_numbers = [
                frame.split("/")[-1].split(".")[0] for frame in frame_paths
            ]

            clip = load_and_preprocess_frames(frame_paths)
            frame_captions = get_frame_captions(frame_paths)
            clip_caption = extract_phase_and_actions(frame_captions)
            """
            dataset.append(
                {
                    "video": video_folder,
                    "frame_numbers": frame_numbers,
                    "clip": clip,
                    "frame_captions": frame_captions,
                    "clip_caption": clip_caption,
                }
            ) """

            start_frame_idx += clip_length - overlap

    return dataset


def main():
    start_video = 0
    end_video = 1
    clip_length = 32
    overlap = 16

    dataset = create_clip_dataset(
        raw_video_dir, clip_length, overlap, start_video, end_video
    )


if __name__ == "__main__":
    main()
