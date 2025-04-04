#ifndef __MFE_H__
#define __MFE_H__

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdint.h"
#include "string.h"
#include "fftw3.h"
#include "time.h"
#include "complex.h"
#include "portaudio.h" // Google PortAudio consists in a library that helps with signal audio processing

#define SAMPLE_RATE 15488 // (16 KHz) NDP101 maximum sampling frequency
#define AUDIO_WINDOW 0.032 // 32 ms
#define DURATION_SECONDS 1.01 // Duration of the audio signal (123 intervals)
#define NUM_SAMPLES ((int)(SAMPLE_RATE * AUDIO_WINDOW)) // Total number of samples
#define NUM_CHANNELS 1
#define MAX_FEATURES 1600
#define ALPHA 0.96875
#define FRAME_STEP 128 // 10ms at 16000
#define FRAME_SIZE 384 // 24ms at 16000
#define NUM_FILTERS 40
#define MIN_FREQ 300
#define MAX_FREQ 8000
#define MIN_MFE 0
#define MAX_MFE 15

short audioBuffer[(int)(SAMPLE_RATE * DURATION_SECONDS)]; // Buffer to capture microphone input
short audioFeatures[MAX_FEATURES];
short filterPreEmphasis[(int)(SAMPLE_RATE * DURATION_SECONDS)];
short frame[FRAME_SIZE];
float spectrum[FRAME_SIZE / 2];
float melFilterBank[NUM_FILTERS][FRAME_SIZE / 2];
float melEnergies[NUM_FILTERS];
uint8_t quantizedMFE[MAX_FEATURES];

// Start capture command
void start_sampling();
static int audioCallback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData);
void convertAudiotoMFE();
void pre_emphasis_filter(short *input, short *output, int samples);
void printSignal(short *array, int samples);
void framing();
void windowing(short *frame, int size);
void fft(short *frame, int size, float *spectrum);
float hzToMel(float hz);
float melToHz(float mel);
void melFilterBankCreation();
void melFilterBankApplication(float *spectrum, float *melEnergies);
void applyLogarithm(float *melEnergies);
void quantizeMFE(float *melEnergies, uint8_t *quantizedMFE);
void printQuantizedMFE(uint8_t *quantizedMFE);

#endif