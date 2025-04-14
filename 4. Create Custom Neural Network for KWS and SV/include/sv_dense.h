#ifndef __SV_DENSE_H__
#define __SV_DENSE_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_LAYERS_SV 4
#define INPUT_SIZE_SV 1600 //Input Features from MFE
#define HIDDEN_LAYER_1_SIZE_SV 256//192 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE_SV 256//192 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE_SV 256//192 //Hidden Layer 3
#define OUTPUT_SIZE_SV 256 //Output Layer

int sv_dense_neural_network(const float input[INPUT_SIZE_SV]);

#endif
