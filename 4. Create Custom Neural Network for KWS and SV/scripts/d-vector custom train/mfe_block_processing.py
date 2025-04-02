import os
import shutil
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.signal import get_window
import soundfile as sf
from collections import defaultdict

SAMPLE_RATE = 16000
FRAME_DUR = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DUR)
FRAME_STRIDE_DUR = 0.024
FRAME_STRIDE = int(SAMPLE_RATE * FRAME_STRIDE_DUR)
NUM_BINS = FRAME_SIZE // 2
FILTER_NUMBER = 40
MIN_FREQ = 0
MAX_FREQ = SAMPLE_RATE // 2
COEFFICIENT = 0.96875
NOISE_FLOOR = -40.0

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

def pre_emphasis(audio):
    emphasized = np.zeros_like(audio, dtype=np.float32)
    emphasized[0] = audio[0] / 32768.0
    for i in range(1, len(audio)):
        emphasized[i] = (audio[i] / 32768.0) - COEFFICIENT * (audio[i-1] / 32768.0)
    return emphasized

def apply_windowing(frame):
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(frame)) / (len(frame) - 1))
    return frame * window

def hz_to_mel(hz):
    return 1127.0 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 1127.0) - 1)

def create_mel_filterbank():
    min_mel = hz_to_mel(MIN_FREQ)
    max_mel = hz_to_mel(MAX_FREQ)
    #mel_points = np.linspace(min_mel, max_mel, FILTER_NUMBER + 2)
    #hz_points = mel_to_hz(mel_points)
    mel_points = np.zeros(FILTER_NUMBER + 2)
    mel_spacing = (max_mel - min_mel) / (FILTER_NUMBER + 1)
    for i in range(FILTER_NUMBER + 2):
        mel_points[i] = mel_to_hz(min_mel + i * mel_spacing)
        if mel_points[i] > MAX_FREQ:
            mel_points[i] = MAX_FREQ

    #bin_indices = np.floor((NUM_BINS) * hz_points / (SAMPLE_RATE / 2)).astype(int)
    #bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)
    bin_indices = np.zeros(FILTER_NUMBER + 2, dtype=int)
    for i in range(FILTER_NUMBER + 2):
        bin_indices[i] = int(mel_points[i] * (NUM_BINS - 1) / (SAMPLE_RATE / 2.0))
        bin_indices[i] = max(0, min(NUM_BINS - 1, bin_indices[i]))

    filterbank = np.zeros((FILTER_NUMBER, NUM_BINS))

    for i in range(FILTER_NUMBER):
        left = bin_indices[i]
        middle = bin_indices[i+1]
        right = bin_indices[i+2]

        if left == middle:
            middle = min(left + 1, NUM_BINS - 1)
        if middle == right:
            right = min(middle + 1, NUM_BINS - 1)

        #filterbank[i, left:middle] = np.linspace(0, 1, middle - left)
        for j in range(left, middle):
            filterbank[i, j] = (j - left) / (middle - left)

        #filterbank[i, middle:right] = np.linspace(1, 0, right - middle)
        for j in range(middle, right):
            filterbank[i, j] = 1.0 - (j - middle) / (right - middle)
    return filterbank

def compute_spectrogram(audio, show_plot=False):
    num_samples = len(audio)
    total_duration = num_samples / SAMPLE_RATE
    num_frames_full_second = int((total_duration - FRAME_DUR) / FRAME_STRIDE_DUR) + 1
    num_frames = min(num_frames_full_second, 40)
    pre_emphasis_array = pre_emphasis(audio)
    spectrogram = np.zeros((num_frames, NUM_BINS))

    for frame in range(num_frames):
        start = frame * FRAME_STRIDE
        end = start + FRAME_SIZE
        segment = pre_emphasis_array[start:end]
        if len(segment) < FRAME_SIZE:
            segment = np.pad(segment, (0, FRAME_SIZE - len(segment)))

        windowed = apply_windowing(segment)
        fft = np.fft.rfft(windowed, n=FRAME_SIZE)
        magnitude = np.abs(fft)
        spectrogram[frame] = magnitude[:NUM_BINS]

    mel_filterbank = create_mel_filterbank()
    mel_spectrogram = np.dot(spectrogram, mel_filterbank.T)
    log_mel_spectrogram = 10* np.log10(mel_spectrogram + 1e-20)

    log_mel_spectrogram = (log_mel_spectrogram - NOISE_FLOOR) / (-NOISE_FLOOR + 12)
    log_mel_spectrogram = np.clip(log_mel_spectrogram, 0, 1)
    quantized = np.round(log_mel_spectrogram * 256) / 256.0
    quantized = np.where(quantized >= 0.65, quantized, 0)
    quantized = quantized[:40]

    if show_plot:
        plt.figure(figsize=(10, 6))
        time_axis = np.linspace(0, 0.968, 40)
        plt.imshow(quantized.T, aspect='auto', origin='lower',
                  extent=[0, 0.968, 0, FILTER_NUMBER])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel filter index')
        plt.title('40x40 Mel Spectrogram (0.968s duration)')
        plt.show()

    return quantized

import os
import shutil
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf
from collections import defaultdict

def process_folder_to_npz(folder_path, output_npz_path, target_user_id, remain_folder=None, is_training=False):
    """Process all WAV files in a folder, using remain folder as fallback"""
    features = []
    labels = []
    filenames = []
    remain_files_used = 0

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    remain_files = []
    if remain_folder and os.path.exists(remain_folder):
        if is_training:
            remain_files = [f for f in os.listdir(remain_folder)
                          if f.endswith('.wav') and f.startswith('target_')
                          and os.path.isfile(os.path.join(remain_folder, f))]
        else:
            remain_files = [f for f in os.listdir(remain_folder)
                          if f.endswith('.wav') and os.path.isfile(os.path.join(remain_folder, f))]
        
        random.shuffle(remain_files)

    for wav_file in tqdm(wav_files, desc=f"Processing {os.path.basename(folder_path)}"):
        audio_path = os.path.join(folder_path, wav_file)
        success = False

        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            audio_int16 = (audio * 32767).astype(np.int16)

            if len(audio_int16) == SAMPLE_RATE:
                mfe = compute_spectrogram(audio_int16)
                mfe = mfe[..., np.newaxis]
                features.append(mfe)

                label = target_user_id if wav_file.startswith('target_') else -1
                labels.append(label)
                filenames.append(wav_file)
                success = True
            else:
                print(f"Duration mismatch: {wav_file} has {len(audio_int16)/SAMPLE_RATE:.2f}s (expected 1.0)")
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

        if not success and remain_files:
            remain_file = remain_files.pop()
            remain_path = os.path.join(remain_folder, remain_file)

            try:
                audio, sr = librosa.load(remain_path, sr=SAMPLE_RATE, mono=True)
                audio_int16 = (audio * 32767).astype(np.int16)

                if len(audio_int16) == SAMPLE_RATE:
                    mfe = compute_spectrogram(audio_int16)
                    mfe = mfe[..., np.newaxis]
                    features.append(mfe)

                    if is_training:
                        label = target_user_id
                    else:
                        label = target_user_id if remain_file.startswith('target_') else -1
                    
                    labels.append(label)
                    filenames.append(f"remain_replacement_{remain_file}")
                    remain_files_used += 1
                    success = True
                else:
                    print(f"Remain file duration mismatch: {remain_file}")
            except Exception as e:
                print(f"Error processing remain file {remain_file}: {str(e)}")

        if not success:
            print(f"Could not process {wav_file} and no valid remain files available")

    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)

    np.savez_compressed(
        output_npz_path,
        features=features_array,
        filenames=np.array(filenames),
        labels=labels_array
    )

    print(f"\nSaved {len(features)} segments to {output_npz_path}")
    print(f"Class distribution: Target={np.sum(labels_array == target_user_id)}, Non-target={np.sum(labels_array != target_user_id)}")
    if remain_files_used > 0:
        print(f"Used {remain_files_used} files from remain folder as replacements")

def process_all_folders(base_dir, target_user_id=0):
    """Process all subfolders in the organized directory"""
    subfolders = [
        "train_1",
        "train_8",
        "train_16",
        "train_64",
        "validation",
        "testing"
    ]

    output_dir = os.path.join(base_dir, "npz_features")
    os.makedirs(output_dir, exist_ok=True)

    # Path to remain folder
    remain_folder = os.path.join(base_dir, "remain")

    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            if "train" in folder:
                output_filename = f"{folder}_{target_user_id}_features.npz"
                is_training = True
            else:
                output_filename = f"{folder}_features.npz"
                is_training = False

            output_path = os.path.join(output_dir, output_filename)

            # Process folder with remain folder as fallback
            process_folder_to_npz(
                folder_path,
                output_path,
                target_user_id,
                remain_folder=remain_folder,
                is_training=is_training
            )

if __name__ == "__main__":
    organized_dir = "/content/dataset/user_0_organized"
    target_speaker_id = 0

    print("Verifying folder structure...")
    for folder in ["validation", "testing", "train_1", "train_8", "train_16", "train_64", "remain"]:
        path = os.path.join(organized_dir, folder)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.wav')]
            print(f"{folder}: {len(files)} files")

    process_all_folders(organized_dir, target_user_id=target_speaker_id)
    remove_all_folders_except(parent_dir=organized_dir, folder_to_keep="npz_features")
