#include "input_audio.h"

static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    short* audio=(short*)inputBuffer;
    float spectrogram[NUM_FRAMES(framesPerBuffer)*FILTER_NUMBER];
    //FOR DEBUG
    for(int i=0; i<framesPerBuffer; i++) {
        printf("%d\t", audio[i]);
    }
    printf("\n%lu\n", framesPerBuffer);

    compute_spectrogram(audio, spectrogram, framesPerBuffer);
    dense_neural_network(spectrogram);
    return paContinue;
}

void channel_setup() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio init failed: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    printf("PortAudio initialized successfully\n");
}

PaStreamParameters parameters_setup() {
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    inputParams.channelCount = NUM_CHANNELS;
    inputParams.sampleFormat = paInt16;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = NULL;
    return inputParams;
}

void identify_device(PaStreamParameters inputParams) {
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
}

void setup_stream(PaStream** stream, PaStreamParameters inputParams) {
    PaError err = Pa_OpenStream(
        stream,
        &inputParams,
        NULL,
        SAMPLE_RATE,
        NUM_SAMPLES,
        paClipOff,
        audio_callback,
        NULL
    );
    if (err != paNoError) {
        printf("Failed to open stream: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream setup correctly\n");
}

void start_stream(PaStream* stream) {
    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("Failed to start stream: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream started\n");
}

void stop_stream(PaStream* stream) {
    PaError err = Pa_StopStream(stream);
    if (err != paNoError) {
        printf("Error Stopping Stream: %s", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("STOP RECORDING...\n");
}

void close_stream(PaStream* stream) {
    PaError err = Pa_CloseStream(stream);
    if (err != paNoError) {
        printf("Error Closing Stream: %s", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream closing correctly\n");
}

void terminate_portaudio() {
    Pa_Terminate();
}

int main(int argc, const char* argv[]) {
    channel_setup();
    PaStream* stream=NULL;
    PaStreamParameters inputParams=parameters_setup();
    identify_device(inputParams);
    setup_stream(&stream, inputParams);
    start_stream(stream);
    //while(1) {
        Pa_Sleep(DURATION_SECONDS * 1024);
    //}
    stop_stream(stream);
    close_stream(stream);
    terminate_portaudio();
}