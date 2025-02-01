import torch as t
import PIL.Image as Image
import torchvision.transforms as T
import os
import numpy as np


class Dataset(t.utils.data.Dataset):
    def __init__(self, _dir, transform=None):
        self._dir = _dir
        self.num_classes = 10
        self.transform = transform
        self.img_list = []
        self.label_list = []
        for i in range(self.num_classes):
            path_cur = os.path.join(self._dir, f"{i}")
            img_list_cur = os.listdir(path_cur)
            img_list_cur = [os.path.join(f"{i}", file) for file in img_list_cur]
            self.img_list += img_list_cur
            self.label_list += [i] * len(img_list_cur)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self._dir, self.img_list[idx])
        I_PIL = Image.open(img_path).convert('L')  # Charger l'image en niveaux de gris
    
      
        I = T.ToTensor()(I_PIL)
    
        label = t.tensor(self.label_list[idx], dtype=t.long)  # Récupérer le label
    
        return I, label, img_path  # Retourner l'image transformée, le label et le chemin

