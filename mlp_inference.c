#define STB_IMAGE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stb_image.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 30
#define OUTPUT_SIZE 10

void read_weights_biases(const char *weights_file, const char *biases_file, float *weights, float *biases, int size_in, int size_out) {
    FILE *w_file = fopen(weights_file, "r");
    FILE *b_file = fopen(biases_file, "r");

    if (!w_file || !b_file) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < size_out; i++) {
        biases[i] = fscanf(b_file, "%f", &biases[i]) == 1 ? biases[i] : 0.0f;
        for (int j = 0; j < size_in; j++) {
            weights[i * size_in + j] = fscanf(w_file, "%f", &weights[i * size_in + j]) == 1 ? weights[i * size_in + j] : 0.0f;
        }
    }

    fclose(w_file);
    fclose(b_file);
}

#define MAX(a,b) (((a)>(b))?(a):(b))

void relu(float *tensor, const unsigned int size) 
{
    unsigned int i;
    for (i = 0; i < size; i++) 
    {
        tensor[i] = MAX(tensor[i], 0);
    }
}

void mat_mult(float *input, float *weights, float *output, int input_size, int output_size, float *biases) 
{
    for (int i = 0; i < output_size; i++) 
    {
        output[i] = 0.0f; 
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        output[i] += biases[i]; 
    }
}

void inference(float *input, float *output) 
{
    float fc1_weights[HIDDEN_SIZE * INPUT_SIZE];
    float fc1_biases[HIDDEN_SIZE];
    float fc2_weights[OUTPUT_SIZE * HIDDEN_SIZE];
    float fc2_biases[OUTPUT_SIZE];
    
    read_weights_biases("fc1_weights.txt", "fc1_biases.txt", fc1_weights, fc1_biases, INPUT_SIZE, HIDDEN_SIZE);
    read_weights_biases("fc2_weights.txt", "fc2_biases.txt", fc2_weights, fc2_biases, HIDDEN_SIZE, OUTPUT_SIZE);
    
    float fc1_output[HIDDEN_SIZE];
    mat_mult(input, fc1_weights, fc1_output, INPUT_SIZE, HIDDEN_SIZE, fc1_biases);
    relu(fc1_output, HIDDEN_SIZE);

    mat_mult(fc1_output, fc2_weights, output, HIDDEN_SIZE, OUTPUT_SIZE, fc2_biases);
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    char *image_path = argv[1];

    int width, height, channels;
    unsigned char *image = stbi_load(image_path, &width, &height, &channels, 1);
    if (image == NULL) {
        printf("Error loading image: %s\n", image_path);
        return -1;
    }

    float input[INPUT_SIZE];
    for (int i = 0; i < 28 * 28; i++) {
        input[i] = image[i] / 255.0f;
    }
    stbi_image_free(image);

    float output[OUTPUT_SIZE];
    inference(input, output);

    int predicted_class = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) 
    {
        if (output[i] > output[predicted_class]) {
            predicted_class = i;
        }
    }

    printf("Image: %s -> Classe prédite: %d\n", image_path, predicted_class);
    return 0;
}
