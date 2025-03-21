#include "mfe.h"
#include "dnn.h"

long long int audioBufferIndex = 0;
long long int mfeIndex = 0;

static int audioCallback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *userData) {
    short *input = (short *)inputBuffer;
    for (unsigned long i = 0; i < framesPerBuffer; i++) {
        if (audioBufferIndex < (int)(SAMPLE_RATE * DURATION_SECONDS)) {
            audioBuffer[audioBufferIndex] = input[i];
            printf("%d\t", audioBuffer[audioBufferIndex]);
            audioBufferIndex++;
        } else {
            return paComplete;
        }
    }
    printf("\n");
    return paContinue;  
}

void start_sampling() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio init failed: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    melFilterBankCreation();

    PaStream *stream;
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    inputParams.channelCount = NUM_CHANNELS;
    inputParams.sampleFormat = paInt16;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = NULL;

    const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(inputParams.device);
    if (deviceInfo != NULL) {
        printf("Using input device: %s\n", deviceInfo->name);
        printf("Sample rate: %d\n", SAMPLE_RATE);
        printf("Max input channels: %d\n", deviceInfo->maxInputChannels);
        printf("Default sample rate: %f\n", deviceInfo->defaultSampleRate);
    } else {
        printf("No input device found\n");
        Pa_Terminate();
        exit(1);
    }

    // Open Stream
    err = Pa_OpenStream(
        &stream, // Stream definition
        &inputParams, // Input parameters
        NULL, // Output parameters
        SAMPLE_RATE, // Sample rate
        NUM_SAMPLES, // Number of samples
        paClipOff, // Clip off the output
        audioCallback, // Callback function
        NULL // No callback data
    );
    if (err != paNoError) {
        printf("Failed to open stream: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream setup correctly\n");

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("Failed to start stream: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("STARTING RECORDING...\n");
    Pa_Sleep(DURATION_SECONDS * 1024);
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        printf("Error Stopping Stream: %s", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("STOP RECORDING...\n");

    convertAudiotoMFE();

    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        printf("Error Closing Stream: %s", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream closing correctly\n");
    Pa_Terminate();
    printf("MFE FEATURES - \n\n%lld\n\n", mfeIndex);
}

void convertAudiotoMFE() {
    pre_emphasis_filter(audioBuffer, filterPreEmphasis, (int)(SAMPLE_RATE * DURATION_SECONDS));
    framing();
    printQuantizedMFE(quantizedMFE);
}

void pre_emphasis_filter(short *input, short *output, int samples) {
    // y[n] = x[n] - alpha * x[n-1]
    output[0] = input[0];
    long long int i;
    for (i = 1; i < samples; i++) {
        output[i] = input[i] - ALPHA * input[i - 1];
    }
    printf("SIGNAL TOTAL RAW DATA FEATURES (SAMPLE_RATE * DURATION(s)) - \n\n%lld\n\n", i);
}

void printSignal(short *array, int samples) {
    for (int i = 0; i < samples; i++) {
        printf("%d\t", array[i]);
    }
}

void framing() {
    for (long long int start = 0; start < (int)(SAMPLE_RATE * DURATION_SECONDS); start += FRAME_STEP) {
        long long int end = start + FRAME_SIZE;
        memcpy(frame, &filterPreEmphasis[start], FRAME_SIZE * sizeof(short));
        windowing(frame, FRAME_SIZE);
        fft(frame, FRAME_SIZE, spectrum);
        melFilterBankApplication(spectrum, melEnergies);
        applyLogarithm(melEnergies);
        quantizeMFE(melEnergies, quantizedMFE);
    }
}

void windowing(short *frame, int size) {
    for (int i = 0; i < size; i++) {
        frame[i] *= 0.54 - 0.46 * cos((2 * M_PI * i) / (size - 1));
    }
}

void fft(short *frame, int size, float *spectrum) {
    double input[size];
    fftw_complex *output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
    for (int i = 0; i < size; i++) {
        input[i] = (double)frame[i];
    }
    fftw_plan plan = fftw_plan_dft_r2c_1d(size, input, output, FFTW_ESTIMATE);
    fftw_execute(plan);
    for (int i = 0; i < size / 2; i++) {
        spectrum[i] = sqrt(output[i][0] * output[i][0] + output[i][1] * output[i][1]);
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
}

float hzToMel(float hz) {
    return 1125 * log(1 + hz / 700.0);
}

float melToHz(float mel) {
    return 700 * (exp(mel / 1127.0) - 1);
}

void melFilterBankCreation() {
    float minMel = hzToMel(MIN_FREQ);
    float maxMel = hzToMel(MAX_FREQ);
    printf("minMel: %.6f, maxMel: %.6f\n", minMel, maxMel);
    float mels[NUM_FILTERS + 2];

    for (int i = 0; i < NUM_FILTERS + 2; i++) {
        mels[i] = melToHz(minMel + (maxMel - minMel) * i / (NUM_FILTERS + 1));
        mels[i] = fmax(0.0, floor((FRAME_SIZE + 1) * mels[i] / SAMPLE_RATE));
    }

    for (int i = 1; i <= NUM_FILTERS; i++) {
        for (int j = (int)mels[i - 1]; j <= (int)mels[i + 1]; j++) {
            melFilterBank[i - 1][j] = (j < (int)mels[i]) ?
                (j - mels[i - 1]) / (mels[i] - mels[i - 1]) :
                (mels[i + 1] - j) / (mels[i + 1] - mels[i]);
        }
    }
}

void melFilterBankApplication(float *spectrum, float *melEnergies) {
    for (int i = 0; i < NUM_FILTERS; i++) {
        melEnergies[i] = 0.0;
        for (int j = 0; j < FRAME_SIZE / 2; j++) {
            melEnergies[i] += spectrum[j] * melFilterBank[i][j];
        }
    }
}

void applyLogarithm(float *melEnergies) {
    for (int i = 0; i < NUM_FILTERS; i++) {
        if (melEnergies[i] < 1e-10) { 
            melEnergies[i] = 1e-10;
        }
        melEnergies[i] = log(melEnergies[i]);
    }
}

void quantizeMFE(float *melEnergies, uint8_t *quantizedMFE) {
    for (int i = 0; i < NUM_FILTERS; i++) {
        float scaledValue = (melEnergies[i] - MIN_MFE) * (15.0 / (MAX_MFE - MIN_MFE));
        if (scaledValue < 0) {
            scaledValue = 0;
        } else if (scaledValue > 15) {
            scaledValue = 15;
        }
        quantizedMFE[mfeIndex++] = (uint8_t)round(scaledValue);
    }
}

void printQuantizedMFE(uint8_t *quantizedMFE) {
    for (int i = 0; i < MAX_FEATURES; i++) {
        printf("%d, ", quantizedMFE[i]);
    }
    printf("\n");
}