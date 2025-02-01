#include <stdio.h>
#include <stdlib.h>

int main() {
    char command[512];

    // Parcours des dossiers 0 à 9
    for (int digit = 0; digit < 10; digit++) {
        for (int img_idx = 0; img_idx < 5; img_idx++) {
            // Générer le chemin de l'image
            sprintf(command, "/home/elisa_meriem/Work_mlp/mlp_inference /home/elisa_meriem/Work_mlp/testing_bmp/%d/%d_%d.bmp", digit, digit, img_idx);
            // Exécuter mlp_inference avec le chemin de l'image
            int ret = system(command);
            if (ret != 0) {
                printf("Erreur lors de l'exécution de mlp_inference pour l'image %s\n", command);
            }
        }
    }
    return 0;
}
