#include "sv.h"

void batch_normalization(const float input[], float output[], int height, int width, int num_batch, float gamma, float beta) {
    for(int i=0; i<height*width*num_batch; i++) {
        output[i]=input[i]*gamma+beta; // y=γx+β
    }
}

void conv2d(const float input[], float output[], int in_height, int in_width, int in_channels, int out_channels, int stride, const float weights[], const float biases[])  {
    int out_height=(in_height-3)/stride+1;
    int out_width=(in_width-3)/stride+1;

    for(int oc=0; oc<out_channels; oc++) {
        for(int oh=0; oh<out_height; oh++) {
            for(int ow=0; ow<out_width; ow++) {
                float sum=biases[oc];

                for(int kh=0; kh<3; kh++) {
                    for(int kw=0; kw<3; kw++) {
                        for(int ic=0; ic<in_channels; ic++) {
                            int ih=oh*stride+kh;
                            int iw=ow*stride+kw;
                            if(ih<in_height && iw<in_width) {
                                int input_idx=ih*in_width*in_channels+iw*in_channels+ic;
                                int weight_idx=kh*3*in_channels*out_channels+kw*in_channels*out_channels+ic*out_channels+oc;
                                sum+=input[input_idx]*weights[weight_idx];
                            }
                        }
                    }
                }

                output[oh*out_width*out_channels+ow*out_channels+oc]=max(sum, 0.0f);
            }
        }
    }
}

void max_pool2d(const float input[], float* output, int in_height, int in_width, int channels, int pool_size) {
    int out_height=in_height/pool_size;
    int out_width=in_width/pool_size;

    for(int c=0; c<channels; c++) {
        for(int oh=0; oh<out_height; oh++) {
            for(int ow=0; ow<out_width; ow++) {
                float max_val=-INFINITY;

                for(int ph=0; ph<pool_size; ph++) {
                    for(int pw=0; pw<pool_size; pw++) {
                        int ih=oh*pool_size+ph;
                        int iw=ow*pool_size+pw;
                        if(ih<in_height && iw<in_width) {
                            float val=input[ih*in_width*channels+iw*channels+c];
                            max_val=max(max_val, val);
                        }
                    }
                }

                output[oh*out_width*channels+ow*channels+c]=max_val;
            }
        }
    }
}

//ALREADY FLAT IN THIS C IMPLEMENTATION
/*void flatten(const float input[], float output[], int height, int width, int channels) {
    memcpy(output, input, height*width*channels*sizeof(float));
}*/

void sv_neural_network(const float mfe_input[], float* d_vector_output) {
    printf("\n\nSV NEURAL NETWORK Accessing\n\n");
    float batchNorm[IN_SIZE]; 
    float conv1[CONV_L1_SIZE]; 
    float maxPool1[MAX_POOL_L1_SIZE];
    float conv2[CONV_L2_SIZE];
    float maxPool2[MAX_POOL_L2_SIZE];
    float conv3[CONV_L3_SIZE];
    float conv4[CONV_L4_SIZE];

    batch_normalization(mfe_input, batchNorm, INPUT_H, INPUT_W, INPUT_CHANNELS, batch_norm_mul[0], batch_norm_sub[0]);
    //STRIDE FOR CONVOLUTION (1 - SAME OUTPUT SIZE, 2 - HALF DIMENSION, 3 - A THIRD OF DIMENSION)
    conv2d(batchNorm, conv1, INPUT_H, INPUT_W, INPUT_CHANNELS, CONV_L1_CHANNELS, 1, conv_1_Weights, conv_1_BiasAdd_ReadVariableOp);

    max_pool2d(conv1, maxPool1, CONV_L1_H, CONV_L1_W, CONV_L1_CHANNELS, 3);

    conv2d(maxPool1, conv2, MAX_POOL_L1_H, MAX_POOL_L1_W, MAX_POOL_L1_CHANNELS, CONV_L2_CHANNELS, 1, conv_2_Weights, conv_2_BiasAdd_ReadVariableOp);

    max_pool2d(conv2, maxPool2, CONV_L2_H, CONV_L2_W, CONV_L2_CHANNELS, 2);

    conv2d(maxPool2, conv3, MAX_POOL_L2_H, MAX_POOL_L2_W, MAX_POOL_L2_CHANNELS, CONV_L3_CHANNELS, 2, conv_3_Weights, conv_3_BiasAdd_ReadVariableOp);

    conv2d(conv3, conv4, CONV_L3_H, CONV_L3_W, CONV_L3_CHANNELS, CONV_L4_CHANNELS, 2, conv_4_Weights, conv_4_BiasAdd_ReadVariableOp);

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