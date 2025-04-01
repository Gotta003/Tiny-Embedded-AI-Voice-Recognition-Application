import os
import numpy as np
import librosa
from tqdm import tqdm

# Constants matching your C implementation
SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_LENGTH = 384
N_FFT = 512
N_MELS = 40
PRE_EMPHASIS_COEFF = 0.96785  # From your C code

def load_metadata(input_dir):
    """Load both speaker and chapter metadata"""
    # Load speaker info
    speaker_info = {}
    speaker_file = os.path.join(input_dir, "..", "speaker.txt")  # One level up from train-clean-100
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

    # Load chapter info
    chapter_info = {}
    chapter_file = os.path.join(input_dir, "..", "chapters.txt")  # One level up from train-clean-100
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

def apply_pre_emphasis(y, coeff=PRE_EMPHASIS_COEFF):
    """Apply pre-emphasis matching your C implementation"""
    emphasized = np.zeros_like(y, dtype=np.float32)
    emphasized[0] = y[0]
    for i in range(1, len(y)):
        emphasized[i] = y[i] - coeff * y[i-1]
    return emphasized

def apply_re_emphasis(y, coeff=PRE_EMPHASIS_COEFF):
    """Apply inverse of pre-emphasis (re-emphasis)"""
    reemphasized = np.zeros_like(y, dtype=np.float32)
    reemphasized[0] = y[0]
    for i in range(1, len(y)):
        reemphasized[i] = y[i] + coeff * reemphasized[i-1]
    return reemphasized

def extract_mfe_segmented(audio_path, segment_sec=1.0, target_frames=None):
    """Extract 1-second segmented MFEs (pad/truncate to target_frames if specified)"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = y / np.max(np.abs(y))  # Normalize

    # Pre-emphasis
    y_emphasized = apply_pre_emphasis(y)

    # Calculate segmentation
    hop_samples = HOP_LENGTH
    frames_per_segment = int(segment_sec * SAMPLE_RATE / hop_samples)

    # Process in 1-second chunks
    mfe_segments = []
    for start_idx in range(0, len(y_emphasized), SAMPLE_RATE):  # Jump by 1-second intervals
        end_idx = start_idx + SAMPLE_RATE
        if end_idx > len(y_emphasized):
            break  # Discard incomplete segment

        # Extract segment
        segment = y_emphasized[start_idx:end_idx]

        # Compute STFT
        stft = librosa.stft(
            segment,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=FRAME_SIZE,
            window='hamming'
        )
        spectrogram = np.abs(stft)

        # Convert to dB and normalize
        spectrogram = 10 * np.log10(spectrogram**2 + 1e-20)
        spectrogram = np.maximum(spectrogram, -50)
        spectrogram = (spectrogram + 50) / 62

        # Mel scaling
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=0,
            fmax=8000
        )
        mfe = np.dot(mel_basis, spectrogram)

        # Pad/truncate if target_frames is specified
        if target_frames:
            if mfe.shape[1] < target_frames:
                mfe = np.pad(mfe, ((0, 0), (0, target_frames - mfe.shape[1])))
            else:
                mfe = mfe[:, :target_frames]

        mfe_segments.append(mfe.T)  # Transpose to (time, mel)

    return mfe_segments  # List of (62, 40) arrays (or target_frames if specified)


def process_librispeech_segmented(input_dir, output_path, min_samples=1448, segment_sec=1.0):
    """Process dataset with 1-second segmentation"""
    samples = []
    classes = []
    speaker_info, chapter_info = load_metadata(input_dir)
    speaker_counts = {}

    # First pass: count valid segments per speaker
    print("Counting segments per speaker...")
    for root, _, files in os.walk(input_dir):
        parts = root.split(os.sep)
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            speaker_id = parts[-2]
            for file in files:
                if file.endswith('.flac'):
                    audio_path = os.path.join(root, file)
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                    num_segments = int(len(y) / SAMPLE_RATE)  # Full 1-second segments
                    speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + num_segments

    # Filter speakers with enough segments
    valid_speakers = {spk: cnt for spk, cnt in speaker_counts.items() if cnt >= min_samples}
    speaker_ids = {spk: idx for idx, spk in enumerate(sorted(valid_speakers.keys()))}

    print(f"Processing {len(valid_speakers)} speakers with ≥{min_samples} segments...")

    # Second pass: extract segments
    for speaker in tqdm(valid_speakers):
        processed_segments = 0
        speaker_path = os.path.join(input_dir, speaker)

        for root, _, files in os.walk(speaker_path):
            for file in sorted(files):
                if file.endswith('.flac') and processed_segments < min_samples:
                    audio_path = os.path.join(root, file)
                    mfe_segments = extract_mfe_segmented(audio_path, segment_sec=segment_sec)

                    for mfe in mfe_segments:
                        if processed_segments >= min_samples:
                            break
                        samples.append(mfe)
                        classes.append(speaker_ids[speaker])
                        processed_segments += 1

    # Save as NPZ
    samples = np.array(samples, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)

    np.savez_compressed(
        output_path,
        features=samples,  # Shape: (n_segments, 40, 40)
        speaker_labels=classes
    )
    print(f"Saved {len(samples)} segments (1s each) to {output_path}")


# Run with 1-second segmentation
process_librispeech_segmented(
    input_dir="LibriSpeech/train-clean-100",
    output_path="librispeech-train-100-clean-mfe-1sec.npz",
    min_samples=1448,
    segment_sec=1.0
)
