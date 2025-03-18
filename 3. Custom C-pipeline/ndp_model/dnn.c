#include "dnn.h"
#include "mfcc.h"

int8_t w1[HIDDEN_LAYER_1_SIZE*INPUT_SIZE];
int16_t b1[HIDDEN_LAYER_1_SIZE];

int8_t w2[HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE];
int16_t b2[HIDDEN_LAYER_2_SIZE];

int8_t w3[HIDDEN_LAYER_3_SIZE*HIDDEN_LAYER_2_SIZE];
int16_t b3[HIDDEN_LAYER_3_SIZE];

int8_t w4[OUTPUT_SIZE*HIDDEN_LAYER_3_SIZE];
int16_t b4[OUTPUT_SIZE];
/*float w1[HIDDEN_LAYER_1_SIZE*INPUT_SIZE];
float b1[HIDDEN_LAYER_1_SIZE];

float w2[HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE];
float b2[HIDDEN_LAYER_2_SIZE];

float w3[HIDDEN_LAYER_3_SIZE*HIDDEN_LAYER_2_SIZE];
float b3[HIDDEN_LAYER_3_SIZE];

float w4[OUTPUT_SIZE*HIDDEN_LAYER_3_SIZE];
float b4[OUTPUT_SIZE];*/

/*const float input_scale = 0.003507965710014105f;
const int32_t input_zero_point = -128;
const float output_scale = 0.00390625f;
const int32_t output_zero_point = -128;*/

float weight_scale[NUM_LAYERS] = {0.0010731631191447377f, 0.0019459010800346732f, 0.0013865033397451043f, 0.002070141490548849f};
int32_t weight_zero_point[NUM_LAYERS] = {0,0,0,0};          
float bias_scale[NUM_LAYERS] = {3.7646193504770054e-06, 3.346194716868922e-05, 2.5274990548496135e-05, 6.192741420818493e-05};      
int32_t bias_zero_point[NUM_LAYERS] = {0,0,0,0};     
int layer=0;

float relu(float x) {
    if(x<0.0) {
        return 0.0;
    }
    return x;
}

float dequantize_int8(int8_t value, float scale, int32_t zero_point) {
    return (value - zero_point) * scale;
}

float dequantize_int16(int16_t value, float scale, int32_t zero_point) {
    return (value - zero_point) * scale;
}

float dequantize_int32(int32_t value, float scale, int32_t zero_point) {
    return (value - zero_point) * scale;
}

// z_j=∑_i(\omega_ij*x_i)+b_j
/*void fully_connected_layer(float input[], int8_t weights[], int16_t biases[], float output[],  int input_size, int output_size) {
    for (int j = 0; j < output_size; j++) {
        float z = 0.0;
        for (int i = 0; i < input_size; i++) {
            z += weights[j * input_size + i] * input[i];
        }
        z += biases[j];
        output[j] = relu(z);
    }
}*/

void fully_connected_layer(uint8_t input[], int8_t weights[], int16_t biases[], float output[], int input_size, int output_size, float weight_scale, int32_t weight_zero_point, float bias_scale, int32_t bias_zero_point) {
    for (int j = 0; j < output_size; j++) {
        float z = 0.0;
        for (int i = 0; i < input_size; i++) {
            float w = dequantize_int8(weights[j * input_size + i], weight_scale, weight_zero_point);
            z += w * input[i];
        }
        float b = dequantize_int16(biases[j], bias_scale, bias_zero_point);
        z += b;
        output[j] = relu(z);
    }
    layer++;
}

void intermediate_fully_connected_layer(float input[], int8_t weights[], int16_t biases[], float output[], int input_size, int output_size, float weight_scale, int32_t weight_zero_point, float bias_scale, int32_t bias_zero_point) {
    for (int j = 0; j < output_size; j++) {
        float z = 0.0;
        for (int i = 0; i < input_size; i++) {
            float w = dequantize_int8(weights[j * input_size + i], weight_scale, weight_zero_point);
            z += w * input[i];
        }
        float b = dequantize_int16(biases[j], bias_scale, bias_zero_point);
        z += b;
        output[j] = relu(z);
    }
    layer++;
}


void softmax(float* input, int size) {
    float max = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        printf("input[%d] = %f\n", i, input[i]);
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    for (size_t i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void loadWeights(const char* filename, int8_t array_upload[], int size_upload) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }
    uint8_t current_byte = 0;
    int nibble_position = 0;

    for (int i = 0; i < size_upload; i++) {
        if (nibble_position == 0) {
            size_t bytes_read = fread(&current_byte, sizeof(uint8_t), 1, file);
            if (bytes_read != 1) {
                perror("Failed to read weight");
                fclose(file);
                exit(1);
            }
        }
        if (nibble_position == 0) {
            array_upload[i] = (current_byte >> 4) & 0x0F;
        } else {
            array_upload[i] = current_byte & 0x0F;
        }
        if (array_upload[i] & 0x08) { 
            array_upload[i] |= 0xF0; 
        }

        nibble_position = 1 - nibble_position;
    }
    for (int i = 0; i < 10; i++) {
        printf("array_upload[%d] = %d\n", i, array_upload[i]);
    }

    fclose(file);
}

/*void loadWeights(const char* filename, int8_t array_upload[], int size_upload) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }
    for (int i = 0; i < size_upload; i++) {
        size_t bytes_read = fread(&array_upload[i], sizeof(int8_t), 1, file);
        if (bytes_read != 1) {
            perror("Failed to read weight");
            fclose(file);
            exit(1);
        }
    }

    for(int i=0; i<10; i++) {
        printf("array_weights[%d] = %hhd\n", i, array_upload[i]);
    }

    fclose(file);
}*/

void loadBiases(const char* filename, int16_t array_upload[], int size_upload) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }
    for (int i = 0; i < size_upload; i++) {
        size_t bytes_read = fread(&array_upload[i], sizeof(int16_t), 1, file);
        if (bytes_read != 1) {
            perror("Failed to read bias");
            fclose(file);
            exit(1);
        }
    }

    for(int i=0; i<10; i++) {
        printf("array_biases[%d] = %d\n", i, array_upload[i]);
    }

    fclose(file);
}

void dnn() {
    printf("\nWeights 1");
    loadWeights(FILENAME_W1, w1, HIDDEN_LAYER_1_SIZE * INPUT_SIZE);
    printf("\nBiases 1");
    loadBiases(FILENAME_B1, b1, HIDDEN_LAYER_1_SIZE);
    printf("\nWeights 2");
    loadWeights(FILENAME_W2, w2, HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE);
    printf("\nBiases 2");
    loadBiases(FILENAME_B2, b2, HIDDEN_LAYER_2_SIZE);
    printf("\nWeights 3");
    loadWeights(FILENAME_W3, w3, HIDDEN_LAYER_3_SIZE * HIDDEN_LAYER_2_SIZE);
    printf("\nBiases 3");
    loadBiases(FILENAME_B3, b3, HIDDEN_LAYER_3_SIZE);
    printf("\nWeights 4");
    loadWeights(FILENAME_W4, w4, OUTPUT_SIZE * HIDDEN_LAYER_3_SIZE);
    printf("\nBiases 4");
    loadBiases(FILENAME_B4, b4, OUTPUT_SIZE);

    float fc1[HIDDEN_LAYER_1_SIZE];
    float fc2[HIDDEN_LAYER_2_SIZE];
    float fc3[HIDDEN_LAYER_3_SIZE];
    float output[OUTPUT_SIZE];

    fully_connected_layer(quantizedMFCC, w1, b1, fc1, INPUT_SIZE, HIDDEN_LAYER_1_SIZE, weight_scale[layer], weight_zero_point[layer], bias_scale[layer], bias_zero_point[layer]);
    for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
        fc1[i] = relu(fc1[i]);
    }
    intermediate_fully_connected_layer(fc1, w2, b2, fc2, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, weight_scale[layer], weight_zero_point[layer], bias_scale[layer], bias_zero_point[layer]);
    for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
        fc2[i] = relu(fc2[i]);
    }
    intermediate_fully_connected_layer(fc2, w3, b3, fc3, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, weight_scale[layer], weight_zero_point[layer], bias_scale[layer], bias_zero_point[layer]);
    for (int i = 0; i < HIDDEN_LAYER_3_SIZE; i++) {
        fc3[i] = relu(fc3[i]);
        printf("fc1[%d]=%f\t fc2[%d]=%f\t fc3[%d]=%f\n", i, fc1[i], i, fc2[i], i, fc3[i]);
    }
    intermediate_fully_connected_layer(fc3, w4, b4, output, HIDDEN_LAYER_3_SIZE, OUTPUT_SIZE, weight_scale[layer], weight_zero_point[layer], bias_scale[layer], bias_zero_point[layer]);
    softmax(output, OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Output[%d] = %f\n", i, output[i]);
    }
}

/*void dnn() {
    loadWeights(FILENAME_W1, w1, HIDDEN_LAYER_1_SIZE*INPUT_SIZE);
    loadBiases(FILENAME_B1, b1, HIDDEN_LAYER_1_SIZE);
    loadWeights(FILENAME_W2, w2, HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE);
    loadBiases(FILENAME_B2, b2, HIDDEN_LAYER_2_SIZE);
    loadWeights(FILENAME_W3, w3, HIDDEN_LAYER_3_SIZE*HIDDEN_LAYER_2_SIZE);
    loadBiases(FILENAME_B3, b3, HIDDEN_LAYER_3_SIZE);
    loadWeights(FILENAME_W4, w4, OUTPUT_SIZE*HIDDEN_LAYER_3_SIZE);
    loadBiases(FILENAME_B4, b4, OUTPUT_SIZE);

    float fc1[HIDDEN_LAYER_1_SIZE];
    float fc2[HIDDEN_LAYER_2_SIZE];
    float fc3[HIDDEN_LAYER_2_SIZE];
    float output[OUTPUT_SIZE];

    fully_connected_layer(quantizedMFCC, w1, b1, fc1, INPUT_SIZE, HIDDEN_LAYER_1_SIZE);
    for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
        fc1[i] = relu(fc1[i]);
        //printf("w1[%d]=%hhd\n", i, w1[i]);
    }
    fully_connected_layer(fc1, w2, b2, fc2, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
    for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
        fc2[i] = relu(fc2[i]);
    }
    fully_connected_layer(fc2, w3, b3, fc3, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
    for (int i = 0; i < HIDDEN_LAYER_3_SIZE; i++) {
        fc3[i] = relu(fc3[i]);
        printf("fc1[%d]=%f\t fc2[%d]=%f\t fc3[%d]=%f\n", i, fc1[i], i, fc2[i], i, fc3[i]);
    }
    fully_connected_layer(fc3, w4, b4, output, HIDDEN_LAYER_3_SIZE, OUTPUT_SIZE);
    softmax(output, OUTPUT_SIZE);

    for(int i=0; i<OUTPUT_SIZE; i++) {
        printf("Output[%d]=%f\n", i, output[i]);
    }
}*/
