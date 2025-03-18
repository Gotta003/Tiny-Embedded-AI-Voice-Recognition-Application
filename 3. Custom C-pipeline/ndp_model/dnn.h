#ifndef __DNN_H__
#define __DNN_H__
#include <stdint.h>
#include <stddef.h>
#define NUM_LAYERS 4
#define INPUT_SIZE 1600 //Input Features from MFCC
#define HIDDEN_LAYER_1_SIZE 256 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE 256 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE 256 //Hidden Layer 3
#define OUTPUT_SIZE 4 //Output Layer
#define FILENAME_W1 "./weights/sequential_dense_MatMul.bin"
#define FILENAME_B1 "./weights/sequential_dense_BiasAdd_ReadVariableOp.bin"
#define FILENAME_W2 "./weights/sequential_dense_1_MatMul.bin"
#define FILENAME_B2 "./weights/sequential_dense_1_BiasAdd_ReadVariableOp.bin"
#define FILENAME_W3 "./weights/sequential_dense_2_MatMul.bin"
#define FILENAME_B3 "./weights/sequential_dense_2_BiasAdd_ReadVariableOp.bin"
#define FILENAME_W4 "./weights/sequential_y_pred_MatMul.bin"
#define FILENAME_B4 "./weights/sequential_y_pred_BiasAdd_ReadVariableOp.bin"

extern int8_t w1[HIDDEN_LAYER_1_SIZE*INPUT_SIZE];
extern int16_t b1[HIDDEN_LAYER_1_SIZE];

extern int8_t w2[HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE];
extern int16_t b2[HIDDEN_LAYER_2_SIZE];

extern int8_t w3[HIDDEN_LAYER_3_SIZE*HIDDEN_LAYER_2_SIZE];
extern int16_t b3[HIDDEN_LAYER_3_SIZE];

extern int8_t w4[OUTPUT_SIZE*HIDDEN_LAYER_3_SIZE];
extern int16_t b4[OUTPUT_SIZE];

extern float weight_scale[NUM_LAYERS];
extern int32_t weight_zero_point[NUM_LAYERS];     
extern float bias_scale[NUM_LAYERS];
extern int32_t bias_zero_point[NUM_LAYERS];
extern int layer;

/*extern float w1[HIDDEN_LAYER_1_SIZE*INPUT_SIZE];
extern float b1[HIDDEN_LAYER_1_SIZE];

extern float w2[HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE];
extern float b2[HIDDEN_LAYER_2_SIZE];

extern float w3[HIDDEN_LAYER_3_SIZE*HIDDEN_LAYER_2_SIZE];
extern float b3[HIDDEN_LAYER_3_SIZE];

extern float w4[OUTPUT_SIZE*HIDDEN_LAYER_3_SIZE];
extern float b4[OUTPUT_SIZE];*/

float relu(float x);
void softmax(float *input, int size);
void dnn();
//void fully_connected_layer(float input[], int8_t weights[], int16_t biases[], float output[],  int input_size, int output_size);
void fully_connected_layer(uint8_t input[], int8_t weights[], int16_t biases[], float output[], int input_size, int output_size, float weight_scale, int32_t weight_zero_point, float bias_scale, int32_t bias_zero_point);
void intermediate_fully_connected_layer(float input[], int8_t weights[], int16_t biases[], float output[], int input_size, int output_size, float weight_scale, int32_t weight_zero_point, float bias_scale, int32_t bias_zero_point);
float dequantize_int8(int8_t value, float scale, int32_t zero_point);
float dequantize_int16(int16_t value, float scale, int32_t zero_point);
float dequantize_int32(int32_t value, float scale, int32_t zero_point);
void loadWeights(const char* filename, int8_t array_upload[], int size_upload);
void loadBiases(const char* filename, int16_t array_upload[], int size_upload);

#endif
