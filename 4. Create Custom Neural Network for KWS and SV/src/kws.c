#include "kws.h"
//BLUE - NOISE - RED - UNKNOWN
float relu(float x) {
    return x > 0 ? x : 0;
}
// z_j=∑_i(\omega_ij*x_i)+b_j
void fully_connected_layer(const float input[], float output[],const float weights[], const float biases[], int input_size, int output_size) {
    //int weight_size=input_size*output_size;
    //int biases_size=output_size;
    for(int j=0; j<output_size; j++) {
        float z=0.0f;
        for(int i=0; i<input_size; i++) {
            z+=weights[j*input_size+i]*input[i];
        }
        z+=biases[j];
        output[j]=relu(z);
    }
} 

void softmax(float input[], int size) {
    float max=input[0];
    for(int i=1; i<size; i++) {
        if(input[i]>max) {
            max=input[i];
        }
    }
    float sum=0.0f;
    for(int i=0; i<size; i++) {
        input[i]=exp(input[i]-max);
        sum+=input[i];
    }
    for(int i=0; i<size; i++) {
        input[i]/=sum;
    }
}

int elaborateResult(float output[]) {
    int printOrder[num_classes];
    for (int i=0; i<num_classes; i++) {
        printOrder[i]=i;
    }
    for(int i=0; i<num_classes-1; i++) {
        for(int j=0; j<num_classes-i-1; j++) {
            if(output[j+1]>output[j]) {
                float temp_output = output[j];
                output[j] = output[j+1];
                output[j+1] = temp_output;
                int temp=printOrder[j];
                printOrder[j]=printOrder[j+1];
                printOrder[j+1]=temp;
            }
        }
    }
    for(int i=0; i<OUTPUT_SIZE; i++) {
        printf("%s=%f\n", class_names[printOrder[i]], output[i]);
    }
    return printOrder[0];
}

int kws_neural_network(const float input[INPUT_SIZE]) {
    float fc1[HIDDEN_LAYER_1_SIZE];
    float fc2[HIDDEN_LAYER_2_SIZE];
    float fc3[HIDDEN_LAYER_3_SIZE];
    float output[OUTPUT_SIZE];

    fully_connected_layer(input, fc1, sequential_dense_MatMul, sequential_dense_BiasAdd_ReadVariableOp, INPUT_SIZE, HIDDEN_LAYER_1_SIZE);
    fully_connected_layer(fc1, fc2, sequential_dense_1_MatMul, sequential_dense_1_BiasAdd_ReadVariableOp, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
    fully_connected_layer(fc2, fc3, sequential_dense_2_MatMul, sequential_dense_2_BiasAdd_ReadVariableOp, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
    fully_connected_layer(fc3, output, sequential_y_pred_MatMul, sequential_y_pred_BiasAdd_ReadVariableOp, HIDDEN_LAYER_3_SIZE, OUTPUT_SIZE);

    softmax(output, OUTPUT_SIZE);
    return elaborateResult(output);
}  
