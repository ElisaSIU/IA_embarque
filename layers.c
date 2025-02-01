#include <stdio.h>
#include <stdlib.h>
#include "layers.h"
#include <math.h> 

void conv2d(const float *input, const float *filter, const float *bias, 
            float *output, int in_h, int in_w, int in_c, 
            int out_h, int out_w, int out_c, 
            int kernel_h, int kernel_w, int stride, int padding) {

    // Initialiser la sortie à zéro
    for (int k = 0; k < out_c; k++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                output[k * out_h * out_w + i * out_w + j] = bias[k];
            }
        }
    }

    // Ajouter un padding si nécessaire
    int padded_h = in_h + 2 * padding;
    int padded_w = in_w + 2 * padding;
    float *padded_input = calloc(padded_h * padded_w * in_c, sizeof(float));
    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < in_h; i++) {
            for (int j = 0; j < in_w; j++) {
                padded_input[c * padded_h * padded_w + (i + padding) * padded_w + (j + padding)] =
                    input[c * in_h * in_w + i * in_w + j];
            }
        }
    }

    // Appliquer la convolution
    for (int k = 0; k < out_c; k++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0;
                for (int c = 0; c < in_c; c++) {
                    for (int m = 0; m < kernel_h; m++) {
                        for (int n = 0; n < kernel_w; n++) {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            sum += padded_input[c * padded_h * padded_w + row * padded_w + col] *
                                   filter[k * in_c * kernel_h * kernel_w + c * kernel_h * kernel_w + m * kernel_w + n];
                        }
                    }
                }
                output[k * out_h * out_w + i * out_w + j] += sum;
            }
        }
    }

    free(padded_input);
}



void relu(float *data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}



void max_pooling(const float *input, float *output, 
                 int in_h, int in_w, int in_c, 
                 int pool_h, int pool_w, int stride) {
    int out_h = (in_h - pool_h) / stride + 1;
    int out_w = (in_w - pool_w) / stride + 1;

    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -1e9;
                for (int m = 0; m < pool_h; m++) {
                    for (int n = 0; n < pool_w; n++) {
                        int row = i * stride + m;
                        int col = j * stride + n;
                        float val = input[c * in_h * in_w + row * in_w + col];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[c * out_h * out_w + i * out_w + j] = max_val;
            }
        }
    }
}

void dense_layer(const float *input, const float *weights, const float *biases, 
                 float *output, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = biases[i];
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
    }
}

void softmax(float *input, int num_classes) {
    float max_input = input[0];
    // Trouver la valeur maximale pour la stabilité numérique
    for (int i = 1; i < num_classes; i++) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }

    // Appliquer la transformation exponentielle et calculer la somme
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        input[i] = expf(input[i] - max_input);  // Subtraction for numerical stability
        sum += input[i];
    }

    // Normaliser les sorties pour qu'elles somment à 1
    for (int i = 0; i < num_classes; i++) {
        input[i] /= sum;
    }
}