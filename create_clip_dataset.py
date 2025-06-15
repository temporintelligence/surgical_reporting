import json
import os
import re

from PIL import Image
import torch
from torchvision import transforms


# List of important data as well as saving directories
raw_data_dir = "data/CholecT50"
raw_video_dir = raw_data_dir + "/videos"
raw_annotations_dir = raw_data_dir + "/labels"
save_dir = "data/save"


def load_and_preprocess_frames(clip, target_size=(224, 224), device=None):
    """
    Loads and preprocesses a sequence of image frames.

    Args:
        clip (list of str):
            List of file paths to image frames.
        target_size (tuple of int):
            Desired size (width, height) to resize each frame to.
        device (torch.device or None):
            The device (CPU or GPU) to move the tensors to.
            If None, automatically selects GPU if available.

    Returns:
        torch.Tensor:
            A 4D tensor of shape (F, C, H, W), where:
                - F: number of frames
                - C: number of channels (3 for RGB)
                - H, W: target height and width
    """
    if device is None:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"

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
    """
    Generates textual captions for a list of individual frame image paths.

    Args:
        frame_paths (list of str):
            List of paths to image frames. Each path must include the video folder as its parent.

    Returns:
        list of str:
            List of natural language captions, one for each input frame.
    """
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


def create_sentence(results):
    """
    Converts structured action-phase results into a coherent natural language summary.

    Args:
        results (list of dict):
            List where each dictionary represents a surgical phase and contains:
                - "phase" (str): Name of the surgical phase.
                - "time" (int): Duration of the phase in seconds.
                - "actions" (list of str): List of action descriptions during the phase.

    Returns:
        str:
            A human-readable sentence or paragraph describing the sequence of actions
            across phases, ordered chronologically and joined with natural connectors.
    """
    if not results:
        return ""

    sentence = ""
    phase_count = len(results)
    connectors = ["First", "Then", "Then", "Then", "Finally"]

    for i, entry in enumerate(results):
        phase = entry["phase"]
        time = entry["time"]
        actions = entry["actions"]

        if len(actions) == 1:
            actions_text = actions[0]
        elif len(actions) == 2:
            actions_text = f"{actions[0]} while {actions[1]}"
        else:
            actions_text = f"{', '.join(actions[:-1])} and {actions[-1]}"

        if phase_count == 1:
            sentence += (
                f"During the phase of {phase} lasting {time} seconds, {actions_text}."
            )
        else:
            connector = connectors[min(i, len(connectors) - 1)]
            sentence += f" {connector}, during the phase of {phase} lasting {time} seconds, {actions_text}."

    return sentence.strip()


def extract_phase_and_actions(captions):
    """
    Extracts surgical phases and associated actions from frame-level captions
    and composes a coherent natural language summary.

    Args:
        captions (list of str):
            List of generated captions for individual frames, each containing
            a surgical phase and one or more action descriptions.

    Returns:
        str:
            A human-readable paragraph summarizing the sequence of phases and
            actions over time. Phases are grouped and described with appropriate
            connectors (e.g., "First", "Then").
    """
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
            if action.strip()
            and action.lower()
            != "unknown"  # TODO: check if this is ever the case with unknown
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
    """
    Processes a directory of video frames to generate a dataset of video clips with associated captions.

    Args:
        raw_video_dir (str):
            Path to the directory containing subfolders for each video.
            Each subfolder contains ordered frame images (e.g., PNG or JPEG).

        clip_length (int):
            Number of consecutive frames to include in each clip.

        overlap (int):
            Number of overlapping frames between consecutive clips (for data augmentation or temporal continuity).

        start_video (int):
            Index of the first video to process (used to subset the full set of videos).

        end_video (int):
            Index at which to stop processing videos (exclusive).

    Returns:
        list of dict:
            A list of dictionaries, one per clip, where each dictionary contains:
                - "video": the name of the video folder
                - "frame_numbers": list of frame numbers (as strings) in the clip
                - "clip": a tensor of shape (clip_length, C, H, W) containing the preprocessed frames
                - "frame_captions": list of generated captions for each frame
                - "clip_caption": a single natural language description summarizing the clip
    """
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

            dataset.append(
                {
                    "video": video_folder,
                    "frame_numbers": frame_numbers,
                    "clip": clip,
                    "frame_captions": frame_captions,
                    "clip_caption": clip_caption,
                }
            )

            start_frame_idx += clip_length - overlap

    return dataset


def main():
    start_video = 0
    end_video = 10
    clip_length = 32
    overlap = 16

    dataset = create_clip_dataset(
        raw_video_dir, clip_length, overlap, start_video, end_video
    )

    torch.save(
        dataset, f"{save_dir}/datasets/clip_dataset_{start_video}_{end_video-1}.pt"
    )


if __name__ == "__main__":
    main()
