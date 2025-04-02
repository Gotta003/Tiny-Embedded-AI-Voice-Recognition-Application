import os
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.signal import get_window

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

def load_metadata(input_dir):
    speaker_info = {}
    speaker_file = os.path.join(input_dir, "..", "speaker.txt")
    if os.path.exists(speaker_file):
        with open(speaker_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = [p.strip() for p in line.split('|')]
                    speaker_id = parts[0]
                    speaker_info[speaker_id] = {
                        'sex': parts[1],
                        'subset': parts[2],
                        'minutes': float(parts[3]),
                        'name': parts[4]
                    }

    chapter_info = {}
    chapter_file = os.path.join(input_dir, "..", "chapters.txt")
    if os.path.exists(chapter_file):
        with open(chapter_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = [p.strip() for p in line.split('|')]
                    chapter_id = parts[0]
                    chapter_info[chapter_id] = {
                        'reader': parts[1],
                        'minutes': float(parts[2]),
                        'subset': parts[3],
                        'project_id': parts[4],
                        'book_id': parts[5],
                        'chapter_title': parts[6],
                        'project_title': parts[7]
                    }

    return speaker_info, chapter_info

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
    mel_points = np.linspace(min_mel, max_mel, FILTER_NUMBER + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_indices = np.floor((NUM_BINS) * hz_points / (SAMPLE_RATE / 2)).astype(int)
    bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)
    
    filterbank = np.zeros((FILTER_NUMBER, NUM_BINS))
    
    for i in range(FILTER_NUMBER):
        left = bin_indices[i]
        middle = bin_indices[i+1]
        right = bin_indices[i+2]
        
        if left == middle:
            middle = min(left + 1, NUM_BINS - 1)
        if middle == right:
            right = min(middle + 1, NUM_BINS - 1)
        
        filterbank[i, left:middle] = np.linspace(0, 1, middle - left)
        filterbank[i, middle:right] = np.linspace(1, 0, right - middle)
    
    return filterbank

def compute_spectrogram(audio, show_plot=True):
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
    log_mel_spectrogram = 10 * np.log10(mel_spectrogram + 1e-20)

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

def process_librispeech_segmented(input_dir, output_path, min_samples=1448, segment_sec=1.0):
    """Process dataset with 1-second segmentation"""
    samples = []
    classes = []
    speaker_info, chapter_info = load_metadata(input_dir)
    speaker_counts = {}

    print("Counting segments per speaker...")
    for root, _, files in os.walk(input_dir):
        parts = root.split(os.sep)
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            speaker_id = parts[-2]
            for file in files:
                if file.endswith('.flac'):
                    audio_path = os.path.join(root, file)
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                    num_segments = int(len(y) / SAMPLE_RATE)
                    speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + num_segments

    valid_speakers = {spk: cnt for spk, cnt in speaker_counts.items() if cnt >= min_samples}
    speaker_ids = {spk: idx for idx, spk in enumerate(sorted(valid_speakers.keys()))}

    print(f"Processing {len(valid_speakers)} speakers with ≥{min_samples} segments...")

    for speaker in tqdm(valid_speakers):
        processed_segments = 0
        speaker_path = os.path.join(input_dir, speaker)

        for root, _, files in os.walk(speaker_path):
            for file in sorted(files):
                if file.endswith('.flac') and processed_segments < min_samples:
                    audio_path = os.path.join(root, file)
                    mfe_segments = compute_spectrogram(audio_path, segment_sec=segment_sec)

                    for mfe in mfe_segments:
                        if processed_segments >= min_samples:
                            break
                        samples.append(mfe)
                        classes.append(speaker_ids[speaker])
                        processed_segments += 1

    samples = np.array(samples, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)

    np.savez_compressed(
        output_path,
        features=samples,
        speaker_labels=classes
    )
    print(f"Saved {len(samples)} segments (1s each) to {output_path}")

process_librispeech_segmented(
    input_dir="LibriSpeech/train-clean-100",
    output_path="librispeech-train-100-clean-mfe-1sec.npz",
    min_samples=1448,
    segment_sec=1.0
)
