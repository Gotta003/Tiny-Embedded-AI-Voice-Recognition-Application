#include "input_audio.h"
#include "audio_samples.h"
#include "sv.h"

static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    short* audio=(short*)inputBuffer;
    float spectrogram[NUM_FRAMES(framesPerBuffer)*FILTER_NUMBER];
    framesPerBuffer*=(1.0-AUDIO_WINDOW);
    for(int i=0; i<framesPerBuffer; i++) {
        printf("%d\t", audio[i]);
    }
    audio_processing(audio, framesPerBuffer);
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

void audio_processing(short* audio_buffer, int framesNumber) {
    float output[1600];
    compute_spectrogram(audio_buffer, output, framesNumber);
    int result=kws_neural_network(output);
    //if(strcmp(class_names[result], "sheila")==0) {
        if(sv_neural_network(output)==0) {
            printf("\n\nHELLO MATTEO\n\n");
        }
        else {
            printf("\n\nUSER NOT ENROLLED\n\n");
        }
    /*}
    else {
        printf("\n\nNOT SHEILA WORD RECOGNIZED\n\n");
    }*/
}

void process_wav_file(const char* filename) {
    int framesPerBuffer = SAMPLE_RATE * (DURATION_SECONDS - AUDIO_WINDOW);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening WAV file\n");
        exit(1);
    }
    short* audio_buffer=(short*)malloc(sizeof(short)*framesPerBuffer);
    // Read and print WAV header (first 44 bytes)
    uint8_t header[44];
    fread(header, 1, sizeof(header), file);

    uint16_t audio_format = *(uint16_t*)(header + 20);
    uint16_t bits_per_sample = *(uint16_t*)(header + 34);
    uint16_t num_channels = *(uint16_t*)(header + 22);

    for (int i = 0; i < framesPerBuffer; i++) {
        int16_t bytes[4]; // Enough for 32-bit samples (stereo 16-bit or mono 32-bit)
        size_t bytes_read = fread(bytes, 1, num_channels * (bits_per_sample / 8), file);
        if (bytes_read != num_channels * (bits_per_sample / 8)) {
            printf("End of file reached\n");
            break;
        }
        audio_buffer[i]=*(short*)bytes;
        //printf("%d\t", audio_buffer[i]);
    }
    //printf("\n\n");
    fclose(file);
    audio_processing(audio_buffer, framesPerBuffer);
    free(audio_buffer);
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
        audio_processing(audio_sample, framesPerBuffer);
    }
    else {
        printf("Invalid mode. Use 0 for live sampling or 1 for file processing\n");
        return 1;
    }

    return 0;
}
