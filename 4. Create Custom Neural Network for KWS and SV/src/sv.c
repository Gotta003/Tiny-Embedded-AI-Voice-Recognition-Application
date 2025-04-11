#include "sv.h"

int floor_div(int a, int b) {
    return (a - ((a % b) + b) % b) / b;
}

void save_debug_output_sv(const char* filename, const char* message, float* data, int rows, int cols) {
    FILE* file = fopen(filename, "a");
    if (!file) {
        printf("Error: Could not open debug file for writing\n");
        return;
    }
    fprintf(file, "%s\n", message);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.6f\t", data[i * cols + j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

float relu_sv(float x) {
    return (x>0) ? x : 0.0;
}

void batch_normalization(const float input[], float output[], int height, int width, int num_batch, float gamma, float beta) {
    for(int i=0; i<height*width*num_batch; i++) {
        output[i]=input[i]*gamma+beta; // y=γx+β + or -?
    }
    //save_debug_output_sv("debug.txt", "BatchNormalization:", output, 1, height*width*num_batch);
}

void conv2d(const float input[], float output[], int in_height, int in_width, int in_channels, int out_channels, int kernel_size, int stride, const float weights[], const float biases[], PaddingType padding) {
    int out_height;
    int out_width;
    int pad_top=0;
    int pad_bottom=0;
    int pad_left=0;
    int pad_right=0;

    //"same"
    if(padding==PADDING_SAME) {
        out_height = floor_div(in_height + stride - 1, stride); 
        out_width = floor_div(in_width + stride - 1, stride);
        int pad_h=(out_height-1)*stride+kernel_size-in_height;
        int pad_w=(out_width-1)*stride+kernel_size-in_width;
        pad_h=pad_h>0 ? pad_h : 0;
        pad_w=pad_w>0 ? pad_w : 0;
        pad_top=floor_div(pad_h,2);
        pad_bottom=pad_h-pad_top;
        pad_left=floor_div(pad_w,2);
        pad_right=pad_w-pad_left;
    }
    //"valid"
    else { 
        out_height=floor_div(in_height-kernel_size, stride)+1;
        out_width=floor_div(in_width-kernel_size, stride)+1;
    }
   
    int padded_height=in_height+pad_top+pad_bottom;
    int padded_width=in_width+pad_left+pad_right;
    float* padded_input=(float*)calloc(padded_height*padded_width*in_channels, sizeof(float));

    for(int h=0; h<in_height; h++) {
        for(int w=0; w<in_width; w++) {
            for(int c=0; c<in_channels; c++) {
                int padded_h=h+pad_top;
                int padded_w=w+pad_left;
                const int in_idx = (h * in_width + w) * in_channels + c;
                const int pad_idx = (padded_h * padded_width + padded_w) * in_channels + c;
                padded_input[pad_idx] = input[in_idx];
            }
        }
    }

    for(int i=0; i<out_height; i++) {
        for(int j=0; j<out_width; j++) {
            for(int oc=0; oc<out_channels; oc++) {
                float sum=biases[oc];
                int h_s=i*stride;
                int w_s=j*stride;
                for(int kh=0; kh<kernel_size; kh++) {
                    for(int kw=0; kw<kernel_size; kw++) {
                        int h=h_s+kh;
                        int w=w_s+kw;
                        if(h<padded_height && w<padded_width) {
                            for(int ic=0; ic<in_channels; ic++) {
                                int input_idx=(h*padded_width+w)*in_channels+ic;
                                int weight_idx=((oc*kernel_size+kh)*kernel_size+kw)*in_channels+ic;
                                sum+=padded_input[input_idx]*weights[weight_idx];
                            }
                        }
                    }
                }
                output[(i*out_width+j)*out_channels+oc]=relu_sv(sum);
            }
        }
    }
    //save_debug_output_sv("debug.txt", "CONV2D:", output, 1, out_height*out_width*out_channels);
    free(padded_input);

}

void max_pool2d(const float input[], float* output, int in_height, int in_width, int channels, int pool_size, int stride, PaddingType padding) {
    int out_height, out_width;
    int pad_top=0, pad_bottom=0, pad_left=0, pad_right=0;
    if (padding==PADDING_VALID) {
        out_height = floor_div(in_height - pool_size, stride)+1;
        out_width = floor_div(in_width - pool_size, stride)+1;
    } 
    else if (padding==PADDING_SAME) {
        out_height = floor_div(in_height+stride-1, stride);
        out_width = floor_div(in_width+stride-1, stride);
        
        int pad_needed_height = (out_height - 1) * stride + pool_size - in_height;
        int pad_needed_width = (out_width - 1) * stride + pool_size - in_width;
        
        pad_top = floor_div(pad_needed_height, 2);
        pad_bottom = pad_needed_height - pad_top;
        pad_left = floor_div(pad_needed_width, 2);
        pad_right = pad_needed_width - pad_left;
    }
    for (int i=0; i<channels*out_height*out_width; i++) {
        output[i]=0.0f;
    }
    for(int c=0; c<channels; c++) {
        for(int h=0; h<out_height; h++) {
            for(int w=0; w<out_width; w++) {
                float max_val=-INFINITY;
                int h_s=h*stride-pad_top;
                int w_s=w*stride-pad_left;
                for(int kh=0; kh<pool_size; kh++) {
                    for(int kw=0; kw<pool_size; kw++) {
                        int h_in=h_s+kh;
                        int w_in=w_s+kw;
                        if(h_in>=0 && h_in<in_height && w_in>=0 && w_in<in_width) {
                            int input_idx=((h_in * in_width + w_in) * channels) + c;
                            float val=input[input_idx];
                            if (val > max_val) {
                                max_val = val;
                            } 
                        }
                    }
                }
                output[(h*out_width+w)*channels+c]=max_val;
            }
        }
    }  
   //save_debug_output_sv("debug.txt", "MAXPOOL:", output, 1, channels*out_height*out_width);
}

//ALREADY FLAT IN THIS C IMPLEMENTATION
/*void flatten(const float input[], float output[], int height, int width, int channels) {
    memcpy(output, input, height*width*channels*sizeof(float));
}*/

float cosine_similarity(const float vec1[DVECTORS], const float vec2[DVECTORS]) {
    float dot_product=0.0f;
    float norm_vec1=0.0f;
    float norm_vec2=0.0f;
    for(int i=0; i<DVECTORS; i++) {
        dot_product+=vec1[i]*vec2[i];
        norm_vec1+=vec1[i]*vec1[i];
        norm_vec2+=vec2[i]*vec2[i];
    }
    norm_vec1=sqrtf(norm_vec1);
    norm_vec2=sqrtf(norm_vec2);
    if(norm_vec1==0 || norm_vec2==0) {
        return 0.0f;
    }
    return dot_product/(norm_vec1*norm_vec2);
}

void normalize_vector(float vector[], int size) {
    float min=vector[0];
    float max=vector[0];
    for(int i=1; i<size; i++) {
        if(vector[i]<min) {min=vector[i];}
        if(vector[i]>max) {max=vector[i];}
    }
    float range=max-min;
    if(range==0) return;
    for(int i=0; i<size; i++) {
        vector[i]=(vector[i]-min)/range;
    }
}

float compute_similarity(const float input_vector[DVECTORS], const float d_vectors[][DVECTORS], int num_vectors) {
    float max_similarity=-1.0f;
    for(int i=0; i<num_vectors; i++) {
        float similarity=cosine_similarity(input_vector, d_vectors[i]);
        if(similarity>max_similarity) {
            max_similarity=similarity;
        }
    }
    return max_similarity;
}

void bestmatching(const float input_vectors[][DVECTORS], const float d_vectors[][DVECTORS], float y_prediction_prob[], int num_inputs, int vector_size) {
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=compute_similarity(input_vectors[i], d_vectors, vector_size);
    }
}

int sv_neural_network(const float mfe_input[]) {
    //printf("\n\nSV NEURAL NETWORK Accessing\n\n");
    float batchNorm[IN_SIZE]; 
    float conv1[CONV_L1_SIZE]; 
    float maxPool1[MAX_POOL_L1_SIZE];
    float conv2[CONV_L2_SIZE];
    float maxPool2[MAX_POOL_L2_SIZE];
    float conv3[CONV_L3_SIZE];
    float conv4[CONV_L4_SIZE];
    int kernel_size=3;

    batch_normalization(mfe_input, batchNorm, INPUT_H, INPUT_W, INPUT_CHANNELS, batch_norm_mul[0], batch_norm_sub[0]);
    //STRIDE FOR CONVOLUTION (1 - SAME OUTPUT SIZE, 2 - HALF DIMENSION, 3 - A THIRD OF DIMENSION)
    conv2d(batchNorm, conv1, INPUT_H, INPUT_W, INPUT_CHANNELS, CONV_L1_CHANNELS, kernel_size, 1, conv_1_Weights, conv_1_BiasAdd_ReadVariableOp, PADDING_SAME);
    
    max_pool2d(conv1, maxPool1, CONV_L1_H, CONV_L1_W, CONV_L1_CHANNELS, 3, 3, PADDING_VALID);
 
    conv2d(maxPool1, conv2, MAX_POOL_L1_H, MAX_POOL_L1_W, MAX_POOL_L1_CHANNELS, CONV_L2_CHANNELS, kernel_size, 1, conv_2_Weights, conv_2_BiasAdd_ReadVariableOp, PADDING_SAME);

    max_pool2d(conv2, maxPool2, CONV_L2_H, CONV_L2_W, CONV_L2_CHANNELS, 2, 2, PADDING_VALID);
    
    conv2d(maxPool2, conv3, MAX_POOL_L2_H, MAX_POOL_L2_W, MAX_POOL_L2_CHANNELS, CONV_L3_CHANNELS, kernel_size, 2, conv_3_Weights, conv_3_BiasAdd_ReadVariableOp, PADDING_SAME);
   
    conv2d(conv3, conv4, CONV_L3_H, CONV_L3_W, CONV_L3_CHANNELS, CONV_L4_CHANNELS, kernel_size, 2, conv_4_Weights, conv_4_BiasAdd_ReadVariableOp, PADDING_SAME);

    int num_inputs=1;
    float prob_0[num_inputs];
    float input_vectors[1][DVECTORS];
    memcpy(input_vectors[0], conv4, sizeof(float) * DVECTORS);
    /*int cols=8;
    for(int i=0; i<DVECTORS; i++) {
        printf("%.6f\t", input_vectors[0][i]);
        if(i%8==7) {
            printf("\n");
        }
    }*/
    //TEST
    bestmatching(input_vectors, d_vectors_0_16, prob_0, num_inputs, 16);
    printf("PROB 0: %.6f", prob_0[0]);
    return (prob_0[0]>SIMILARITY_THRESHOLD ? 0 : 1);
}

/*Model: "d-vector-extractor-256"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ batch_normalization                  │ (None, 40, 40, 1)           │               4 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 40, 40, 8)           │              80 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 8)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 13, 13, 16)          │           1,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 6, 6, 16)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 3, 3, 32)            │           4,640 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 2, 2, 64)            │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 24,388 (95.27 KB)
 Trainable params: 24,386 (95.26 KB)    
 Non-trainable params: 2 (8.00 B)*/
