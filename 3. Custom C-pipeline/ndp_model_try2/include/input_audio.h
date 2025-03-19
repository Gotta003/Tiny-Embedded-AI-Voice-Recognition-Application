#ifndef __INPUT_AUDIO_H__
#define __INPUT_AUDIO_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <portaudio.h>
#include "spectrogram.h"
#include "dense_neural_network.h"

#define SAMPLE_RATE 16000
#define AUDIO_WINDOW 0.032
#define DURATION_SECONDS 1
#define NUM_CHANNELS 1
#define NUM_SAMPLES (int)(SAMPLE_RATE * (DURATION_SECONDS))
#define FFT_SIZE 512

static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData);
void channel_setup();
PaStreamParameters parameters_setup();
void identify_device(PaStreamParameters inputParams);
void setup_stream(PaStream** stream, PaStreamParameters inputParams);
void start_stream(PaStream* stream);
void stop_stream(PaStream* stream);
void close_stream(PaStream* stream);
void terminate_portaudio();
void live_sampling();
void test_sampling();
#endif
