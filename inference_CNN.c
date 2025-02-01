#include <stdio.h>
#include <stdlib.h>
#include "layers.h"
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <dirent.h>
#include <ctype.h> 


// Fonction pour charger une image
unsigned char* load_image(const char *filename, int *width, int *height, int *channels) {
    unsigned char *data = stbi_load(filename, width, height, channels, 0);
    if (data == NULL) {
        fprintf(stderr, "Erreur de chargement de l'image %s\n", filename);
        exit(1);
    }
    return data;
}

// Redimensionnement de l'image
unsigned char* resize_image(unsigned char *data, int width, int height, int desired_width, int desired_height, int channels) {
    unsigned char *resized_data = malloc(desired_width * desired_height * channels);
    if (resized_data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour l'image redimensionnée.\n");
        exit(1);
    }
    stbir_resize_uint8_srgb(data, width, height, 0, resized_data, desired_width, desired_height, 0, channels);
    return resized_data;
}

// Normalisation de l'image : [0, 255] -> [0, 1]
void normalize_image(unsigned char *data, float *normalized_data, int size) {
    for (int i = 0; i < size; i++) {
        normalized_data[i] = data[i] / 255.0f; // Normalisation
    }
}

// Fonction pour charger les images depuis un dossier
void load_images_from_folder(const char *folder_path, int label, 
                              float **images, int *labels, int *count, int image_size) {
    DIR *dir = opendir(folder_path);
    struct dirent *entry;

    if (dir == NULL) {
        perror("Erreur lors de l'ouverture du dossier");
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { // Vérifier si c'est un fichier
            char filepath[1024];
            snprintf(filepath, sizeof(filepath), "%s/%s", folder_path, entry->d_name);

            int width, height, channels;
            unsigned char *image_data = load_image(filepath, &width, &height, &channels);

            // Redimensionner l'image à la taille souhaitée (par exemple 28x28)
            int desired_width = 28, desired_height = 28;
            unsigned char *resized_data = resize_image(image_data, width, height, desired_width, desired_height, channels);

            // Normalisation
            int resized_image_size = desired_width * desired_height * channels;
            float *normalized_data = malloc(resized_image_size * sizeof(float));
            normalize_image(resized_data, normalized_data, resized_image_size);

            // Ajouter l'image et l'étiquette à la liste
            images[*count] = normalized_data;
            labels[*count] = label;
            (*count)++;

            free(image_data);
            free(resized_data);
        }
    }

    closedir(dir);
}

// Fonction pour charger toutes les images depuis les dossiers
void load_all_images(const char *dataset_path, float ***images, int **labels, int *total_images) {
    DIR *dir = opendir(dataset_path);
    struct dirent *entry;

    if (dir == NULL) {
        perror("Erreur lors de l'ouverture du dossier dataset");
        return;
    }

    // Compter le nombre total d'images pour allouer la mémoire
    *total_images = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && entry->d_name[0] != '.') { // Ignorer "." et ".."
            char folder_path[1024];
            snprintf(folder_path, sizeof(folder_path), "%s/%s", dataset_path, entry->d_name);
            DIR *subdir = opendir(folder_path);
            struct dirent *sub_entry;

            while ((sub_entry = readdir(subdir)) != NULL) {
                if (sub_entry->d_type == DT_REG) {
                    (*total_images)++;
                }
            }
            closedir(subdir);
        }
    }

    // Allouer de la mémoire pour les images et les labels
    *images = malloc(*total_images * sizeof(float*));
    *labels = malloc(*total_images * sizeof(int));

    // Revenir au début du dossier
    rewinddir(dir);

    // Charger toutes les images
    int count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && entry->d_name[0] != '.') { // Ignorer "." et ".."
            char folder_path[1024];
            snprintf(folder_path, sizeof(folder_path), "%s/%s", dataset_path, entry->d_name);

            // Charger les images du dossier correspondant à la classe (label)
            load_images_from_folder(folder_path, atoi(entry->d_name), *images, *labels, &count, 28 * 28 * 3); // Taille de l'image à 28x28 pixels et 3 canaux RGB
        }
    }

    closedir(dir);
}


float* load_weights(const char* filename, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erreur : Impossible d'ouvrir le fichier %s\n", filename);
        return NULL;
    }

    // Lire le fichier une première fois pour compter le nombre de valeurs
    int count = 0;
    float temp;
    while (fscanf(file, "%f", &temp) == 1) {  // Lire chaque nombre flottant, un par ligne
        count++;
    }

    if (count == 0) {
        printf("Erreur : Aucun poids trouvé dans %s\n", filename);
        fclose(file);
        return NULL;
    }

    rewind(file);  // Retour au début du fichier

    // Allouer un tableau pour stocker les poids
    float* weights = (float*)malloc(count * sizeof(float));
    if (!weights) {
        printf("Erreur d'allocation mémoire\n");
        fclose(file);
        return NULL;
    }

    // Lire les valeurs et les stocker dans le tableau
    for (int i = 0; i < count; i++) {
        if (fscanf(file, "%f", &weights[i]) != 1) {  // Lire chaque nombre flottant
            printf("Erreur de lecture dans %s à l'index %d\n", filename, i);
            free(weights);
            fclose(file);
            return NULL;
        }
    }
    fclose(file);
    
    *size = count;  // Stocker la taille du tableau
    return weights;
}

int main() {

    //Charger les poids
    int conv1_w_size=32*3*3*3, conv1_b_size=32, conv2_w_size=64*32*3*3, conv2_b_size=64;
    int fc1_w_size=64*128*7*7, fc1_b_size=128, fc2_w_size=128*10, fc2_b_size=10;
    
    float *conv1_weights = load_weights("model_param/conv1_weights.txt", &conv1_w_size);
    float *conv1_biases  = load_weights("model_param/conv1_biases.txt", &conv1_b_size);
    float *conv2_weights = load_weights("model_param/conv2_weights.txt", &conv2_w_size);
    float *conv2_biases  = load_weights("model_param/conv2_biases.txt", &conv2_b_size);
    float *fc1_weights   = load_weights("model_param/fc1_weights.txt", &fc1_w_size);
    float *fc1_biases    = load_weights("model_param/fc1_biases.txt", &fc1_b_size);
    float *fc2_weights   = load_weights("model_param/fc2_weights.txt", &fc2_w_size);
    float *fc2_biases    = load_weights("model_param/fc2_biases.txt", &fc2_b_size);
    

    printf("Premier poids de layer1: %f\n", conv1_weights[8]);
    printf("Dernier poids de layer1: %f\n", conv1_weights[conv1_w_size - 5]);
    printf("size: %d, %d, %d, %d\n", conv1_w_size, conv1_b_size, conv2_w_size, fc1_b_size);

    //Charger les images
    float **images = NULL;
    int *labels = NULL;
    int total_images = 0;

    const char *dataset_path = "bdd_chiffre/testing"; 
    load_all_images(dataset_path, &images, &labels, &total_images);

    // Affichage pour vérifier
    printf("Total d'images chargées : %d\n", total_images);

 
    const int num_classes = 10;
    int correct_predictions = 0;

    //Décrire réseau
    for (int i = 0; i < total_images; i++) {
        float *input = images[i];
        //Couche 1
        float conv1_output[32 * 14 * 14];
        conv2d(input, conv1_weights, conv1_biases, conv1_output, 
            28, 28, 3, 14, 14, 32, 3, 3, 1, 1);
        relu(conv1_output, 32 * 14 * 14);

        float pool1_output[32 * 7 * 7];
        max_pooling(conv1_output, pool1_output, 14, 14, 32, 2, 2, 2);

        // Couche 2
        float conv2_output[64 * 7 * 7];
        conv2d(pool1_output, conv2_weights, conv2_biases, conv2_output, 
            7, 7, 32, 7, 7, 64, 3, 3, 1, 1);
        relu(conv2_output, 64 * 7 * 7);

        float pool2_output[64 * 3 * 3];
        max_pooling(conv2_output, pool2_output, 7, 7, 64, 2, 2, 2);

        // Couches FC
        float fc1_output[128];
        dense_layer(pool2_output, fc1_weights, fc1_biases, fc1_output, 64 * 3 * 3, 128);
        relu(fc1_output, 128);

        float fc2_output[num_classes];
        dense_layer(fc1_output, fc2_weights, fc2_biases, fc2_output, 128, num_classes);
        softmax(fc2_output, num_classes);

        int predicted_class = 0;
        float max_prob = fc2_output[0];
        for (int j = 1; j < num_classes; j++) {
            if (fc2_output[j] > max_prob) {
                max_prob = fc2_output[j];
                predicted_class = j;
            }
        }
        printf("Résultats d'inférence pour l'image %d (Vraie étiquette : %d) :\n", i, labels[i]);
        printf("Classe prédite %d \n", predicted_class);

        // Vérifier si la prédiction est correcte
        if (predicted_class == labels[i]) {
            correct_predictions++;
        }


    }

    // Calcul et affichage de la précision
    float accuracy = (float)correct_predictions / total_images * 100.0;
    printf("Précision sur l'ensemble des images : %.2f%%\n", accuracy);

    // Libération de la mémoire
    free(conv1_weights);
    free(conv1_biases);
    free(conv2_weights);
    free(conv2_biases);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    
    for (int i = 0; i < total_images; i++) {
        free(images[i]);
    }
    free(images);
    free(labels);

    return 0;
}
