#include "kws.h"
#include "sv_dense.h"
#include "sv.h"
#include "d_vector_dense_extractor.h"
#include "d_vectors_dense.h"

int sv_dense_neural_network(const float input[INPUT_SIZE_SV]) {
    float fc1[HIDDEN_LAYER_1_SIZE_SV];
    float fc2[HIDDEN_LAYER_2_SIZE_SV];
    float fc3[HIDDEN_LAYER_3_SIZE_SV];
    float output[OUTPUT_SIZE_SV];

    fully_connected_layer(input, fc1, sequential_dense_1_MatMul_SV, sequential_dense_1_BiasAdd_ReadVariableOp_SV, INPUT_SIZE_SV, HIDDEN_LAYER_1_SIZE_SV);
    fully_connected_layer(fc1, fc2, sequential_dense_2_MatMul_SV, sequential_dense_2_BiasAdd_ReadVariableOp_SV, HIDDEN_LAYER_1_SIZE_SV, HIDDEN_LAYER_2_SIZE_SV);
    fully_connected_layer(fc2, fc3, sequential_dense_3_MatMul_SV, sequential_dense_3_BiasAdd_ReadVariableOp_SV, HIDDEN_LAYER_2_SIZE_SV, HIDDEN_LAYER_3_SIZE_SV);
    fully_connected_layer(fc3, output, sequential_dense_4_MatMul_SV, sequential_dense_4_BiasAdd_ReadVariableOp_SV, HIDDEN_LAYER_3_SIZE_SV, OUTPUT_SIZE_SV);

    int num_inputs=1;
    float prob_0[num_inputs];
    float input_vectors[1][DVECTORS];
    memcpy(input_vectors[0], output, sizeof(float) * DVECTORS);
    /*int cols=8;
    for(int i=0; i<DVECTORS; i++) {
        printf("%.6f\t", input_vectors[0][i]);
        if(i%8==7) {
            printf("\n");
        }
    }*/

    bestmatching(input_vectors, d_vectors_0_16_SV, prob_0, num_inputs, 16);
    printf("PROB 0: %.6f", prob_0[0]);
    return (prob_0[0]>SIMILARITY_THRESHOLD ? 0 : 1);
}