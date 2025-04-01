import os
import shutil
import random
from tqdm import tqdm

def organize_wav_files(target_source_dir, other_source_dir, output_dir, target_user_id=0):
    """
    Organizes wav files into structured folders for speaker verification.
    
    Args:
        target_source_dir: Directory containing target speaker's wav files (user_0/)
        other_source_dir: Directory containing other speakers' wav files (other_users/)
        output_dir: Output directory for organized files
        target_user_id: Numeric ID for target speaker
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory structure
    subfolders = {
        "validation": (60, 120),    # (target_samples, non_target_samples)
        "testing": (60, 120),
        "train_1": (1, 0),
        "train_8": (8, 0),
        "train_16": (16, 0),
        "train_64": (64, 0)
    }

    # Create all folders
    for folder in subfolders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Load target speaker files
    target_files = [f for f in os.listdir(target_source_dir)
                   if f.endswith('.wav') and os.path.isfile(os.path.join(target_source_dir, f))]
    random.shuffle(target_files)
    
    # Load non-target speaker files
    other_files = []
    for root, _, files in os.walk(other_source_dir):
        for file in files:
            if file.endswith('.wav'):
                other_files.append(os.path.join(root, file))
    random.shuffle(other_files)

    # Verify we have enough files
    required_target = sum(v[0] for v in subfolders.values())
    required_other = sum(v[1] for v in subfolders.values())
    
    if len(target_files) < required_target:
        raise ValueError(f"Need {required_target} target files, found {len(target_files)}")
    if len(other_files) < required_other:
        raise ValueError(f"Need {required_other} non-target files, found {len(other_files)}")

    # Organize files
    target_idx = 0
    other_idx = 0
    
    for folder, (target_count, other_count) in subfolders.items():
        # Move target speaker files
        for i in range(target_count):
            if target_idx >= len(target_files):
                break
            src = os.path.join(target_source_dir, target_files[target_idx])
            dst = os.path.join(output_dir, folder, f"target_{target_files[target_idx]}")
            shutil.copy(src, dst)
            target_idx += 1
        
        # Move non-target speaker files
        for i in range(other_count):
            if other_idx >= len(other_files):
                break
            src = other_files[other_idx]
            dst = os.path.join(output_dir, folder, f"other_{os.path.basename(src)}")
            shutil.copy(src, dst)
            other_idx += 1

    print("Organization complete.")
    print(f"Used {target_idx} target files and {other_idx} non-target files")
    print(f"Remaining target files: {len(target_files) - target_idx}")
    print(f"Remaining non-target files: {len(other_files) - other_idx}")

if __name__ == "__main__":
    # Configure these paths
    target_speaker_dir = "/content/dataset/user_0/"  # Contains only target speaker
    other_speakers_dir = "/content/dataset/others/"  # Contains other speakers
    output_directory = "/content/dataset/user_0_organized/"
    
    organize_wav_files(
        target_source_dir=target_speaker_dir,
        other_source_dir=other_speakers_dir,
        output_dir=output_directory,
        target_user_id=0
    )
    !rm -r /content/dataset/user_0
    !rm -r /content/dataset/others
