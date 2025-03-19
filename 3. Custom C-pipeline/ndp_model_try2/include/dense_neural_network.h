#ifndef __DENSE_NEURAL_NETWORK_H__
#define __DENSE_NEURAL_NETWORK_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model_weights.h"
#define NUM_LAYERS 4
#define INPUT_SIZE 1600 //Input Features from MFCC
#define HIDDEN_LAYER_1_SIZE 256 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE 256 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE 256 //Hidden Layer 3
#define OUTPUT_SIZE 4 //Output Layer

float relu(float x);
void fully_connected_layer(const float input[], float output[],const float weights[], const float biases[], int input_size, int output_size);
void softmax(float input[], int size);
void dense_neural_network(const float input[INPUT_SIZE]);

#endif
