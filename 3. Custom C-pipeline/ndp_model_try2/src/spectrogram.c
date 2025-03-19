#include "spectrogram.h"

void pre_emphasis(short* audio, float pre_emphasis_array[], unsigned long num_samples) {
    pre_emphasis_array[0] = (float)audio[0];
    for(int i=1; i<num_samples; i++) {
        pre_emphasis_array[i] = (float)audio[i] - COEFFICIENT * (float)audio[i-1];
        //DEBUG
        //printf("%d\t%hd\t%f\n", i, audio[i], pre_emphasis_array[i]);
    }
}

void apply_windowing(kiss_fft_cpx* frame, int size) {
    for(int i=0; i<size; i++) {
        frame[i].r *= 0.54 - 0.46 * cos(2 * M_PI * i / (size - 1));
        frame[i].i *= 0.54 - 0.46 * cos(2 * M_PI * i / (size - 1));
    }
}

void spectrogram_population(kiss_fft_cpx fft_out[], float spectrogram[], int frame) {
   for(int i=0; i<NUM_BINS; i++) {
        spectrogram[frame*NUM_BINS+i] = sqrt(fft_out[i].r*fft_out[i].r + fft_out[i].i*fft_out[i].i);
        //DEBUG
        //printf("%d\t%f\n", i, spectrogram[frame*(FRAME_SIZE/2+1)+i]);
   }
}

void framing_operation(float* pre_emphasis_audio, float spectrogram[], unsigned long num_samples) {
    kiss_fft_cfg cfg=kiss_fft_alloc(FRAME_SIZE, 0, NULL, NULL);
    kiss_fft_cpx fft_in[FRAME_SIZE], fft_out[FRAME_SIZE];

    for(int frame=0; frame<NUM_FRAMES(num_samples); frame++) {
        int start=frame*FRAME_STRIDE;
        for(int i=0; i<FRAME_SIZE; i++) {
            fft_in[i].r=pre_emphasis_audio[start+i];
            fft_in[i].i=0.0f;
        }
        apply_windowing(fft_in, FRAME_SIZE);
        kiss_fft(cfg, fft_in, fft_out);
        spectrogram_population(fft_out, spectrogram, frame);
    }
    free(cfg);
}

float hz_to_mel (float hz) {
    return 2595*log10(1+hz/700);
}

float mel_to_hz (float mel) {
    return 700*(pow(10, mel/2595)-1);
}

void create_mel_filterbank(float mel_filterbank[FILTER_NUMBER][NUM_BINS]) {
    float min_mel=hz_to_mel(MIN_FREQ);
    float max_mel=hz_to_mel(MAX_FREQ);
    float mel_points[FILTER_NUMBER + 2];
    for (int i = 0; i < FILTER_NUMBER + 2; i++) {
        mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (FILTER_NUMBER+1));
    }
    int fft_bins[FILTER_NUMBER+2];
    for (int i = 0; i < FILTER_NUMBER + 2; i++) {
        fft_bins[i] = floor((FRAME_SIZE + 1) * mel_points[i] / SAMPLE_RATE);
    }
    for (int i = 0; i < FILTER_NUMBER; i++) {
        for (int j = 0; j < NUM_BINS; j++) {
            if (j < fft_bins[i]) {
                mel_filterbank[i][j] = 0;
            } else if (j < fft_bins[i + 1]) {
                mel_filterbank[i][j] = (j - fft_bins[i]) / (fft_bins[i + 1] - fft_bins[i]);
            } else if (j < fft_bins[i + 2]) {
                mel_filterbank[i][j] = (fft_bins[i + 2] - j) / (fft_bins[i + 2] - fft_bins[i + 1]);
            } else {
                mel_filterbank[i][j] = 0;
            }
        }
    }
}

void apply_mel_filterbank(float spectrogram[], float mel_filterbank[FILTER_NUMBER][NUM_BINS], float log_mel_spectrogram[], int num_frames) {
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < FILTER_NUMBER; j++) {
            float sum = 0;
            for (int k = 0; k < NUM_BINS; k++) {
                sum += spectrogram[i*NUM_BINS+k] * mel_filterbank[j][k];
            }
            log_mel_spectrogram[i*FILTER_NUMBER+j] = log10(sum+1e-10);
        }
    }
}

void save_spectrogram(float log_mel_spectrogram[], int num_frames, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file for writing\n");
        return;
    }

    for (int frame = 0; frame < num_frames; frame++) {
        for (int i = 0; i < FILTER_NUMBER; i++) {
            fprintf(file, "%.6f ", log_mel_spectrogram[frame * FILTER_NUMBER+ i]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void mean_filter(float log_mel_spectrogram[], int num_frames) {
    for(int i=0; i<num_frames; i++) {
        for(int j=0; j<FILTER_NUMBER; j++)) {
            float sum=0.0;
            int count=0;
            float value=log_mel_spectrogram[i*FILTER_NUMBER+j];
            for(int di=-1; di<=1; di++) {
                for(int dj=-1; dj=1; dj++) {
                    int ni=i+di;
                    int nj=j+dj;
                    if(ni>=0 && ni<num_frames && nj>=0 && nj<FILTER_NUMBER) {
                        sum+=log_mel_spectrogram[ni*FILTER_NUMBER+nj];
                        count++;
                    }
                }
            }
            log_mel_spectrogram[i*FILTER_NUMBER+j]=sum/count;
            //DEBUG
            printf("%f\t%f\n", value, log_mel_spectrogram[i*FILTER_NUMBER+j]);
        }
    }
}

void apply_noise_floor(float log_mel_spectrogram[], int num_frames) {
    for(int i=0; i<num_frames; i++) {
        for(int j=0; j<FILTER_NUMBER; j++) {
            if(log_mel_spectrogram[i*FILTER_NUMBER+j]<NOISE_FLOOR) {
                log_mel_spectrogram[i*FILTER_NUMBER+j]=0.0f;
            }
            //printf("%d\t%d\t%d\t%f\n", i*FILTER_NUMBER+j, i, j, log_mel_spectrogram[i*FILTER_NUMBER+j]);
        }
    }
}

void compute_spectrogram(short* audio, float log_mel_spectrogram[], unsigned long num_samples) {
    float pre_emphasis_array[num_samples];
    pre_emphasis(audio, pre_emphasis_array, num_samples);
    float spectrogram[NUM_FRAMES(num_samples) * NUM_BINS];
    framing_operation(pre_emphasis_array, spectrogram, num_samples);
    float mel_filterbank[FILTER_NUMBER][NUM_BINS];
    create_mel_filterbank(mel_filterbank);
    apply_mel_filterbank(spectrogram, mel_filterbank, log_mel_spectrogram, NUM_FRAMES(num_samples));
    mean_filter(log_mel_spectrogram, NUM_FRAMES(num_samples));
    apply_noise_floor(log_mel_spectrogram, NUM_FRAMES(num_samples));
    save_spectrogram(log_mel_spectrogram, NUM_FRAMES(num_samples), SPECTROGRAM_FILENAME);
}