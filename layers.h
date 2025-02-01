#ifndef LAYERS_H
#define LAYERS_H

void conv2d(const float *input, const float *filter, const float *bias, 
            float *output, int in_h, int in_w, int in_c, 
            int out_h, int out_w, int out_c, 
            int kernel_h, int kernel_w, int stride, int padding);

void relu(float *data, int size);

void max_pooling(const float *input, float *output, 
                 int in_h, int in_w, int in_c, 
                 int pool_h, int pool_w, int stride);

void dense_layer(const float *input, const float *weights, const float *biases, 
                 float *output, int input_size, int output_size);

void softmax(float *data, int size);

#endif