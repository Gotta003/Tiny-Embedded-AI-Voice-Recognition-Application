import os
import shutil
import random
from tqdm import tqdm

def organize_wav_files(target_source_dir, other_source_dir, output_dir, target_user_id=0):

    os.makedirs(output_dir, exist_ok=True)
    subfolders = {
        "validation": (60, 120),
        "testing": (60, 120),
        "train_1": (1, 0),
        "train_8": (8, 0),
        "train_16": (16, 0),
        "train_64": (64, 0),
        "remain": (-1, -1)
    }

    for folder in subfolders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    target_files = [f for f in os.listdir(target_source_dir)
                   if f.endswith('.wav') and os.path.isfile(os.path.join(target_source_dir, f))]
    random.shuffle(target_files)
    
    other_files = []
    for root, _, files in os.walk(other_source_dir):
        for file in files:
            if file.endswith('.wav'):
                other_files.append(os.path.join(root, file))
    random.shuffle(other_files)

    required_target = sum(v[0] for k, v in subfolders.items() if k != "remain")
    required_other = sum(v[1] for k, v in subfolders.items() if k != "remain")
    
    if len(target_files) < required_target:
        raise ValueError(f"Need {required_target} target files, found {len(target_files)}")
    if len(other_files) < required_other:
        raise ValueError(f"Need {required_other} non-target files, found {len(other_files)}")

    target_idx = 0
    other_idx = 0
    
    for folder, (target_count, other_count) in subfolders.items():
        if folder == "remain":
            continue

        for i in range(target_count):
            src = os.path.join(target_source_dir, target_files[target_idx])
            dst = os.path.join(output_dir, folder, f"target_{target_files[target_idx]}")
            shutil.copy(src, dst)
            target_idx += 1

        for i in range(other_count):
            src = other_files[other_idx]
            dst = os.path.join(output_dir, folder, f"other_{os.path.basename(src)}")
            shutil.copy(src, dst)
            other_idx += 1

    for remaining_target in target_files[target_idx:]:
        src = os.path.join(target_source_dir, remaining_target)
        dst = os.path.join(output_dir, "remain", f"target_{remaining_target}")
        shutil.copy(src, dst)

    for remaining_other in other_files[other_idx:]:
        dst = os.path.join(output_dir, "remain", f"other_{os.path.basename(remaining_other)}")
        shutil.copy(remaining_other, dst)

    used_target = target_idx
    used_other = other_idx
    remaining_target = len(target_files) - target_idx
    remaining_other = len(other_files) - other_idx

    print("\nOrganization complete.")
    print(f"Used {used_target} target files and {used_other} non-target files")
    print(f"Remaining target files moved to 'remain': {remaining_target}")
    print(f"Remaining non-target files moved to 'remain': {remaining_other}")
    print("\nFinal counts per folder:")
    for folder in subfolders:
        if folder == "remain":
            target_count = len([f for f in os.listdir(os.path.join(output_dir, folder))
                             if f.startswith('target_')])
            other_count = len([f for f in os.listdir(os.path.join(output_dir, folder))
                            if f.startswith('other_')])
            print(f"{folder}: {target_count} target, {other_count} non-target (remaining)")
        else:
            expected_target, expected_other = subfolders[folder]
            actual_target = len([f for f in os.listdir(os.path.join(output_dir, folder))
                              if f.startswith('target_')])
            actual_other = len([f for f in os.listdir(os.path.join(output_dir, folder))
                             if f.startswith('other_')])
            print(f"{folder}: {actual_target}/{expected_target} target, {actual_other}/{expected_other} non-target")

if __name__ == "__main__":
    target_speaker_dir = "/content/dataset/user_0/"
    other_speakers_dir = "/content/dataset/others/"
    output_directory = "/content/dataset/user_0_organized/"
    
    organize_wav_files(
        target_source_dir=target_speaker_dir,
        other_source_dir=other_speakers_dir,
        output_dir=output_directory,
        target_user_id=0
    )
    
    !rm -r /content/dataset/user_0
    !rm -r /content/dataset/others
