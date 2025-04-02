#ifndef __SV_H__
#define __SV_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "d_vector_extractor.h"
#include "d_vectors.h"

#define INPUT_H 40
#define INPUT_W 40
#define IN_SIZE (INPUT_H*INPUT_W)
#define INPUT_CHANNELS 1

#define CONV_L1_H (INPUT_H)
#define CONV_L1_W (INPUT_W)
#define CONV_L1_CHANNELS 8
#define CONV_L1_SIZE (CONV_L1_H*CONV_L1_W*CONV_L1_CHANNELS) 

#define MAX_POOL_L1_H 13
#define MAX_POOL_L1_W 13
#define MAX_POOL_L1_CHANNELS (CONV_L1_CHANNELS)
#define MAX_POOL_L1_SIZE (MAX_POOL_L1_H*MAX_POOL_L1_W*MAX_POOL_L1_CHANNELS) 

#define CONV_L2_H (MAX_POOL_L1_H)
#define CONV_L2_W (MAX_POOL_L1_W)
#define CONV_L2_CHANNELS (CONV_L1_CHANNELS*2)
#define CONV_L2_SIZE (CONV_L2_H*CONV_L2_W*CONV_L2_CHANNELS) 

#define MAX_POOL_L2_H 6
#define MAX_POOL_L2_W 6
#define MAX_POOL_L2_CHANNELS (CONV_L2_CHANNELS)
#define MAX_POOL_L2_SIZE (MAX_POOL_L2_H*MAX_POOL_L2_W*MAX_POOL_L2_CHANNELS)

#define CONV_L3_H 3
#define CONV_L3_W 3
#define CONV_L3_CHANNELS (CONV_L2_CHANNELS*2)
#define CONV_L3_SIZE (CONV_L3_H*CONV_L3_W*CONV_L3_CHANNELS)

#define CONV_L4_H 2
#define CONV_L4_W 2
#define CONV_L4_CHANNELS (CONV_L3_CHANNELS*2)
#define CONV_L4_SIZE (CONV_L4_H*CONV_L4_W*CONV_L4_CHANNELS)

//#define OUTPUT_SIZE (CONV_L4_SIZE)

#define max(a, b) ((a) > (b) ? (a) : (b))
#define DVECTORS 256

int sv_neural_network(const float mfe_input[]);
void batch_normalization(const float input[], float output[], int height, int width, int num_batch, float gamma, float beta);
void conv2d(const float input[], float output[], int in_height, int in_width, int in_channels, int out_channels, int kernel_size, int stride, const float weights[], const float biases[], const char padding[]) ;
void max_pool2d(const float input[], float* output, int in_height, int in_width, int channels, int pool_size);
//void flatten(const float input[], float output[], int height, int width, int channels);
int bestmatching(const float input_vector[], const float d_vectors[][DVECTORS], int num_d_vectors, const int input_labels[], int num_inputs, int auth_label, float threshold, int verbose);
float cosine_similarity(const float vec1[DVECTORS], const float vec2[DVECTORS]);
void save_debug_output_sv(const char* filename, const char* message, float* data, int rows, int cols);

#endif
