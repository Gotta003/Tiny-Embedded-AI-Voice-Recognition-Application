import os
import shutil
import numpy as np
import librosa
from tqdm import tqdm

SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_LENGTH = 384
N_FFT = 512
N_MELS = 40
PRE_EMPHASIS_COEFF = 0.96785

def remove_all_folders_except(parent_dir, folder_to_keep):
    keep_path = os.path.join(parent_dir, folder_to_keep)
    if not os.path.exists(keep_path):
        print(f"Warning: '{folder_to_keep}' doesn't exist in {parent_dir}")
        return

    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and item != folder_to_keep:
            print(f"Removing: {item_path}")
            try:
                shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to remove {item_path}: {e}")

def apply_pre_emphasis(y, coeff=PRE_EMPHASIS_COEFF):
    emphasized = np.zeros_like(y, dtype=np.float32)
    emphasized[0] = y[0]
    for i in range(1, len(y)):
        emphasized[i] = y[i] - coeff * y[i-1]
    return emphasized

def extract_mfe(waveform):
    emphasized = apply_pre_emphasis(waveform)
    
    stft = librosa.stft(
        emphasized,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE,
        window='hamming'
    )
    spectrogram = np.abs(stft)
    spectrogram = 10 * np.log10(spectrogram**2 + 1e-20)
    spectrogram = np.maximum(spectrogram, -50)
    spectrogram = (spectrogram + 50) / 62
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000
    )
    mfe = np.dot(mel_basis, spectrogram)
    mfe = mfe.T
    mfe = mfe[:40, :]

    return mfe

def process_folder_to_npz(folder_path, output_npz_path, target_user_id):
    features = []
    labels = []
    filenames = []
    
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for wav_file in tqdm(wav_files, desc=f"Processing {os.path.basename(folder_path)}"):
        audio_path = os.path.join(folder_path, wav_file)
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except:
            print(f"Skipping corrupted file: {audio_path}")
            continue
        
        if len(y) != SAMPLE_RATE:
            print(f"Warning: {wav_file} has {len(y)/SAMPLE_RATE:.2f} seconds (expected 1.0)")
            continue
        
        mfe = extract_mfe(y)
        mfe = mfe[..., np.newaxis]
        features.append(mfe)
      
        if wav_file.startswith('target_'):
            labels.append(target_user_id)
        else:
            labels.append(-1)
        
        filenames.append(wav_file)
    
    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)
    
    np.savez_compressed(
        output_npz_path,
        features=features_array,
        filenames=np.array(filenames),
        labels=labels_array
    )
    print(f"Saved {len(features)} segments to {output_npz_path}")
    print(f"Class distribution: Target={np.sum(labels_array == target_user_id)}, Non-target={np.sum(labels_array != target_user_id)}")

def process_all_folders(base_dir, target_user_id=0):
    """Process all subfolders in the organized directory"""
    subfolders = [
        "validation",
        "testing",
        "train_1",
        "train_8",
        "train_16",
        "train_64"
    ]
    
    output_dir = os.path.join(base_dir, "npz_features")
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            if "train" in folder:
                output_filename = f"{folder}_{target_user_id}_features.npz"
            else:
                output_filename = f"{folder}_features.npz"

            output_path = os.path.join(output_dir, output_filename)
            process_folder_to_npz(folder_path, output_path, target_user_id)

if __name__ == "__main__":
    organized_dir = "/content/dataset/user_0_organized"
    target_speaker_id = 0
    
    print("Verifying folder structure...")
    for folder in ["validation", "testing", "train_1", "train_8", "train_16", "train_64"]:
        path = os.path.join(organized_dir, folder)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.wav')]
            print(f"{folder}: {len(files)} files")
    
    process_all_folders(organized_dir, target_user_id=target_speaker_id)
    remove_all_folders_except(parent_dir=organized_dir, folder_to_keep="npz_features")
