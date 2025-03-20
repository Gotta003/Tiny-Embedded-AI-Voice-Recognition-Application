#ifndef __SPECTROGRAM_H__
#define __SPECTROGRAM_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <portaudio.h>
#include "fft/kiss_fft.h"
#include "audio_samples.h"

#define SAMPLE_RATE 16000
#define FRAME_DUR 0.032
#define FRAME_SIZE (int)(SAMPLE_RATE * FRAME_DUR)
#define FRAME_STRIDE_DUR 0.024
#define FRAME_STRIDE (int)(SAMPLE_RATE * FRAME_STRIDE_DUR)
#define NUM_BINS (int)(FRAME_SIZE/2)
#define FFT_LENGTH 512
#define FILTER_NUMBER 40 
#define MIN_FREQ 0
#define MAX_FREQ (int)(SAMPLE_RATE/2)
#define COEFFICIENT 0.96875
#define NUM_FRAMES(size_audio) (int)((size_audio-FRAME_SIZE)/FRAME_STRIDE+1)
#define SPECTROGRAM_FILENAME "spectrogram.txt"
#define NOISE_FLOOR -100.0f

void compute_spectrogram(short* audio, float spectrogram[], unsigned long num_samples);
void pre_emphasis(short* audio, float pre_emphasis[], unsigned long num_samples);
void apply_windowing(kiss_fft_cpx* frame, int size);
void spectrogram_population(kiss_fft_cpx fft_out[], float spectrogram[], int frame);
void framing_operation(float* pre_emphasis_audio, float spectrogram[], unsigned long num_samples);
float hz_to_mel (float hz);
float mel_to_hz (float mel);
void create_mel_filterbank(float mel_filterbank[FILTER_NUMBER][NUM_BINS]);
void apply_mel_filterbank(float spectrogram[], float mel_filterbank[FILTER_NUMBER][NUM_BINS], float log_mel_spectrogram[], int num_frames);
void apply_noise_floor(float log_mel_spectrogram[], int num_frames);
void mean_filter(float log_mel_spectrogram[], int num_frames);

#endif
