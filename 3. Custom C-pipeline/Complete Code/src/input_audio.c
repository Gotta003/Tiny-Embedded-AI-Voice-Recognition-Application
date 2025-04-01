#include "input_audio.h"
#include "audio_samples.h"

static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    short* audio=(short*)inputBuffer;
    float spectrogram[NUM_FRAMES(framesPerBuffer)*FILTER_NUMBER];
    //FOR DEBUG
    framesPerBuffer*=(1.0-AUDIO_WINDOW);
    for(int i=0; i<framesPerBuffer; i++) {
        printf("%d\t", audio[i]);
    }

    compute_spectrogram(audio, spectrogram, framesPerBuffer);
    dense_neural_network(spectrogram);
    printf("\n%lu\n", framesPerBuffer);
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

void live_sampling() {
    channel_setup();
    PaStream* stream=NULL;
    PaStreamParameters inputParams=parameters_setup();
    identify_device(inputParams);
    setup_stream(&stream, inputParams);
    start_stream(stream);
    //while(1) {
    Pa_Sleep((int)(DURATION_SECONDS*1024));
    //}
    stop_stream(stream);
    close_stream(stream);
    terminate_portaudio();
}

void process_wav_file(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening WAV file\n");
        exit(1);
    }

    // Skip WAV header (assuming 44-byte standard header)
    fseek(file, 44, SEEK_SET);
    
    // Read audio data
    uint8_t bytes[SAMPLE_RATE * 4]; // Assuming stereo 16-bit samples
    size_t bytes_read = fread(bytes, 1, sizeof(bytes), file);
    fclose(file);

    // Convert to your format
    int16_t audio_sample[SAMPLE_RATE];
    for (int i = 0, j = 0; i < bytes_read && j < SAMPLE_RATE; i += 4, j++) {
        double left = (double)((bytes[i] & 0xff) | (bytes[i+1] << 8));
        double right = (double)((bytes[i+2] & 0xff) | (bytes[i+3] << 8));
        
        // Validate sample range
        if(left < -32768 || left > 32767 || right < -32768 || right > 32767) {
            printf("Invalid sample at position %d: L=%.0f R=%.0f\n", i, left, right);
            left = 0; right = 0; // Clamp to silence
        }
        audio_sample[j] = (int16_t)((left + right) / 2);
        printf("%d\t", audio_sample[j]);
    }

    // Process the converted audio
    int framesPerBuffer = SAMPLE_RATE * (DURATION_SECONDS - AUDIO_WINDOW);
    float output[1600];
    compute_spectrogram(audio_sample, output, framesPerBuffer);
    dense_neural_network(output);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  For live sampling: %s 0\n", argv[0]);
        printf("  For file processing: %s 1 <audio_file.wav>\n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);
    
    if (mode == 0) {
        // Live sampling mode
        if (argc != 2) {
            printf("Live sampling mode requires exactly 1 argument\n");
            return 1;
        }
        live_sampling();
    }
    else if (mode == 1) {
        // File processing mode
        if (argc != 3) {
            printf("File processing mode requires exactly 2 arguments\n");
            return 1;
        }
        
        const char* filename = argv[2];
        const char* extension = strrchr(filename, '.');
        
        if (!extension || strcmp(extension, ".wav") != 0) {
            printf("Unsupported file format. Please use .wav\n");
            return 1;
        }
        
        process_wav_file(filename);
    }
    else if (mode==2) {
        int framesPerBuffer = SAMPLE_RATE * (DURATION_SECONDS - AUDIO_WINDOW);
        float output[1600];
        compute_spectrogram(audio_sample, output, framesPerBuffer);
        dense_neural_network(output);
    }
    else {
        printf("Invalid mode. Use 0 for live sampling or 1 for file processing\n");
        return 1;
    }

    return 0;
}
