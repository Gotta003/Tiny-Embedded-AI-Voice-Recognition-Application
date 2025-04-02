#include "sv.h"

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

void batch_normalization(const float input[], float output[], int height, int width, int num_batch, float gamma, float beta) {
    for(int i=0; i<height*width*num_batch; i++) {
        output[i]=input[i]*gamma-beta; // y=γx+β
    }
    //save_debug_output_sv("debug.txt", "BatchNormalization:", output, 1, height*width*num_batch);
}

void conv2d(const float input[], float output[], int in_height, int in_width, int in_channels, int out_channels, int kernel_size, int stride, const float weights[], const float biases[], const char padding[])  {
    int out_height;
    int out_width;
    int pad_top=0;
    int pad_bottom=0;
    int pad_left=0;
    int pad_right=0;

    //"same"
    if(strcmp(padding, "same")==0) {
        out_height=(int)ceilf((float)in_height/(float)stride);
        out_width=(int)ceilf((float)in_width/(float)stride);
        int pad_h=(out_height-1)*stride+kernel_size-in_height;
        int pad_w=(out_width-1)*stride+kernel_size-in_width;
        pad_h=pad_h>0 ? pad_h : 0;
        pad_w=pad_w>0 ? pad_w : 0;
        pad_top=pad_h/2;
        pad_bottom=pad_h-pad_top;
        pad_left=pad_w/2;
        pad_right=pad_w-pad_left;
    }
    //"valid"
    else { 
        out_height=(in_height-kernel_size)/stride+1;
        out_width=(in_width-kernel_size)/stride+1;
    }
   
    int padded_height=in_height+pad_top+pad_bottom;
    int padded_width=in_width+pad_left+pad_right;
    float* padded_input=(float*)calloc(padded_height*padded_width*in_channels, sizeof(float));

    for(int h=0; h<in_height; h++) {
        for(int w=0; w<in_width; w++) {
            for(int c=0; c<in_channels; c++) {
                int padded_h=h+pad_top;
                int padded_w=w+pad_left;
                padded_input[(padded_h*padded_width+padded_w)*in_channels+c]=input[(h*in_width+w)*in_channels+c];
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
                                int weights_idx=((kh*kernel_size+kw)*in_channels+ic)*out_channels+oc;
                                sum+=padded_input[input_idx]*weights[weights_idx];
                            }
                        }
                    }
                }
                output[(i*out_width+j)*out_channels+oc]=sum;
            }
        }
    }
    //save_debug_output_sv("debug.txt", "CONV2D:", output, 1, out_height*out_width*out_channels);
    free(padded_input);

}

void max_pool2d(const float input[], float* output, int in_height, int in_width, int channels, int pool_size) {
    int out_height=(int)(in_height/pool_size);
    int out_width=(int)(in_width/pool_size);

    for(int i = 0; i < out_height * out_width * channels; i++) {
        output[i] = 0.0f;
    }

    for(int i=0; i<out_height; i++) {
        for(int j=0; j<out_width; j++) {
            int h_s=i*pool_size;
            int w_s=j*pool_size;
            for(int c=0; c<channels; c++) {
                int found=0;
                float max_val=-INFINITY;

                for(int ph=0; ph<pool_size; ph++) {
                    for(int pw=0; pw<pool_size; pw++) {
                        int h=h_s+ph;
                        int w=w_s+pw;
                        if(h<in_height && w<in_width) {
                            float val=input[(h*in_width+w)*channels+c];
                            if(val>max_val) {
                                max_val=val;
                                found=1;
                            }
                        }
                    }
                }
                int output_idx = (i * out_width + j) * channels + c;
                output[output_idx] = found ? max_val : 0.0f;
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

int bestmatching(const float input_vector[], const float d_vectors[][DVECTORS], int num_d_vectors, const int input_labels[], int num_inputs, int auth_label, float threshold, int verbose) {
    int total_auth=0;
    int total_denied=0;

    float min=input_vector[0];
    float max=input_vector[0];

    for(int i=1; i<DVECTORS; i++) {
        if(input_vector[i]<min) {
            min=input_vector[i];
        }
        if(input_vector[i]>max) {
            max=input_vector[i];
        }
    }
    float norm_vector[DVECTORS];
    for(int i=0; i<DVECTORS; i++) {
        norm_vector[i]=(input_vector[i]-min)/(max-min);
    }

    for(int i=0; i<num_inputs; i++) {
        if(input_labels[i]==auth_label) {
            total_auth++;
        }
        else {
            total_denied++;
        }
    }
    int correct_auth=0;
    int correct_denied=0;
    for(int i=0; i<num_inputs; i++) {
        float similarity=compute_similarity(norm_vector, d_vectors, num_d_vectors);
        if(verbose) {
            printf("similarity: %f --- Class: %d\n", similarity, input_labels[i]);
        }
        if(similarity>threshold && input_labels[i]==auth_label) {
            correct_auth++;
            return 0;
        }
        if(similarity<=threshold && input_labels[i]!=auth_label) {
            correct_denied++;
            return 1;
        }
    }
    return 2;
}

int sv_neural_network(const float mfe_input[]) {
    printf("\n\nSV NEURAL NETWORK Accessing\n\n");
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
    conv2d(batchNorm, conv1, INPUT_H, INPUT_W, INPUT_CHANNELS, CONV_L1_CHANNELS, kernel_size, 1, conv_1_Weights, conv_1_BiasAdd_ReadVariableOp, "same");

    max_pool2d(conv1, maxPool1, CONV_L1_H, CONV_L1_W, CONV_L1_CHANNELS, 3);

    conv2d(maxPool1, conv2, MAX_POOL_L1_H, MAX_POOL_L1_W, MAX_POOL_L1_CHANNELS, CONV_L2_CHANNELS, kernel_size, 1, conv_2_Weights, conv_2_BiasAdd_ReadVariableOp, "same");

    max_pool2d(conv2, maxPool2, CONV_L2_H, CONV_L2_W, CONV_L2_CHANNELS, 2);

    conv2d(maxPool2, conv3, MAX_POOL_L2_H, MAX_POOL_L2_W, MAX_POOL_L2_CHANNELS, CONV_L3_CHANNELS, kernel_size, 2, conv_3_Weights, conv_3_BiasAdd_ReadVariableOp, "same");

    conv2d(conv3, conv4, CONV_L3_H, CONV_L3_W, CONV_L3_CHANNELS, CONV_L4_CHANNELS, kernel_size, 2, conv_4_Weights, conv_4_BiasAdd_ReadVariableOp, "same");
    int cols=8;
    for(int i=0; i<CONV_L4_SIZE; i++) {
        if(i%cols==0) {
            printf("%d - ", i/cols+1);
        }
        printf("%.6f\t", conv4[i]);
        if(i%cols==cols-1) {
            printf("\n");
        }
    }

    const int input_labels[]={0, 1};
    return bestmatching(conv4, d_vectors_0_64, 64, input_labels, 1, 0, 0.6, 1);
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