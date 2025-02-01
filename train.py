#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:53:20 2025

@author: meriem
"""
import torch
import json
from MNISTDataset import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


torch.random.manual_seed(0)
transform = T.Compose([
    #T.Resize((28, 28)),  
    T.ToTensor(),  # Convertir en tensor (les valeurs seront entre 0 et 1)                   
])

#path_train = '/home/meriem/Documents/TSI/IA_legere/Work/MNIST/Training'                
#path_train = '/home/meriem/Documents/TSI/IA_legere/Work/training_bmp'                
path_train = '/home/docker/Work/training_bmp'                
training_set = Dataset(path_train, transform=transform)

plt.figure(1)
for i in range(4):
    image, label, _ = training_set[i]
    plt.subplot(1,4,i+1)
    plt.imshow(T.ToPILImage()(image))
    plt.title('True label {}'.format(label))
    
plt.pause(1.)


batch_size = 32
train_loader = DataLoader(dataset = training_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)
images, labels, _ = next(iter(train_loader))

plt.figure(2)
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(T.ToPILImage()(images[i,:,:,:]))
    plt.title('True label {}'.format(labels[i]))
    
plt.pause(1.)

#path_valid = '/home/meriem/Documents/TSI/IA_legere/Work/MNIST/Testing'                
#path_valid = '/home/meriem/Documents/TSI/IA_legere/Work/testing_bmp'                
path_valid = '/home/docker/Work/testing_bmp'                
valid_set = Dataset(path_valid,transform=transform)
valid_loader = DataLoader(dataset = valid_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)

   
input_size = 28*28

class MLP(nn.Module):
    def __init__(self, H, input_size):
        super(MLP, self).__init__()
        
        self.C = 10
        self.D = input_size
        self.H = H
        
        
        self.fc1 = nn.Linear(self.D, self.H) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.H, self.C)  
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self,X):
      #  print(f"Input to forward: {X.shape}")  # Ajoutez une impression ici pour vérifier la taille
    
        X1 = self.fc1(X) #NxH
       # print(f"After fc1: {X1.shape}")  # Vérifiez la taille après fc1
        X2 = self.relu(X1) #NxH
      #  print(f"After ReLU: {X2.shape}")  # Vérifiez la taille après ReLU
        O = self.fc2(X2) #NxC
        O = self.softmax(O)
    
    
        return O
    

def validation(valid_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, _ in valid_loader:
           # print(f"Image batch size: {images.size(0)}, Label batch size: {labels.size(0)}")
            images_vec = images.view(-1, 28*28)
            
            outputs = model(images_vec)
            #print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
            _, predicted = torch.max(outputs.data, 1)
            #print(f"Predicted shape: {predicted.shape}, Labels shape: {labels.shape}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct, total)

#%% HYPERPARAMETERS
H = 30
lr = 1e-2 #learning rate
beta = 0.9 #momentum parameter
n_epoch = 15#number of iterations
input_size = 28*28

model = MLP(H,input_size)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)  
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

num_batch = len(train_loader) #600 batches each containing 100 images = 60000 images

training_loss_v = []
valid_acc_v = []

(correct, total) = validation(valid_loader, model)
print ('Epoch [{}/{}], Valid Acc: {} %'
           .format(0, n_epoch, 100 * correct / total))
valid_acc_v.append(correct / total)

for epoch in range(n_epoch):
    
    loss_tot = 0

    for i, (images, labels,_) in enumerate(train_loader):
        print(images.shape)  # Devrait afficher [16, 1, 28, 28]

        # Reshape images to (batch_size, input_size), actual shape is (batch_size, 1, 28, 28)
        images_vec = images.view(-1, input_size)
        print(images_vec.shape)  # Devrait afficher [16, 784]

            
        #Forward Pass
        O = model.forward(images_vec)
        
        #Compute Loss
        l = criterion(O, labels)
        
        #Print Loss
        loss_tot += l.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}' 
                   .format(epoch+1, n_epoch, i+1, num_batch, l.item()/len(labels)))
    
        #Backward Pass (Compute Gradient)
        optimizer.zero_grad()
        l.backward()
        
        #Update Parameters
        optimizer.step()    
        
    
    (correct, total) = validation(valid_loader, model)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, n_epoch, loss_tot/len(training_set), 100 * correct / total))
    training_loss_v.append(loss_tot/len(training_set))
    valid_acc_v.append(correct / total)
    
# weights = model.state_dict()
# weights_dict = {key: value.cpu().numpy().tolist() for key, value in weights.items()}
# with open("model_weights.json", "w") as f:
#     json.dump(weights_dict, f)
    
# with open("model_weights.bin", "wb") as f:
#     for key, value in model.state_dict().items():
#         np.array(value.cpu().numpy(), dtype=np.float32).tofile(f)

# print("Les poids ont été sauvegardés dans 'model_weights.bin'.")

# Save the weights and biases of the model to text files
def save_weights_and_biases(model):
    # Extracting the weights and biases of the layers
    fc1_weights = model.fc1.weight.detach().numpy()
    fc1_biases = model.fc1.bias.detach().numpy()
    fc2_weights = model.fc2.weight.detach().numpy()
    fc2_biases = model.fc2.bias.detach().numpy()
    
    # Saving to text files
    np.savetxt('fc1_weights.txt', fc1_weights)
    np.savetxt('fc1_biases.txt', fc1_biases)
    np.savetxt('fc2_weights.txt', fc2_weights)
    np.savetxt('fc2_biases.txt', fc2_biases)

# Save the weights and biases
save_weights_and_biases(model)

#%% plot results
plt.figure(3)
plt.clf()
plt.plot(training_loss_v,'r',label='Training loss')
plt.legend()

plt.figure(4)
plt.clf()
plt.plot(valid_acc_v,'g',label='Validation accuracy')
plt.legend()

(images, labels,_) = next(iter(valid_loader))
print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")
images_vec = images.reshape(-1, input_size)
outputs = model(images_vec)
print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
_, predicted = torch.max(outputs.data, 1)
print(f"Predicted shape: {predicted.shape}")
plt.figure(5)
plt.clf()

# Dictionnaire pour suivre les classes déjà affichées
classes_displayed = set()

# Nombre total d'images dans le lot
num_images = images.size(0)

# Parcourir toutes les images
for idx in range(num_images):
    label = labels[idx].item()
    if label not in classes_displayed:  # Vérifier si la classe n'a pas encore été affichée
        image = images[idx, :]
        plt.subplot(2, 5, len(classes_displayed) + 1)  # Créez une grille 2x5 (une pour chaque classe)
        plt.imshow(T.ToPILImage()(image))
        plt.title(f'True {label} / Pred {predicted[idx].item()}')
        classes_displayed.add(label)  # Marquez la classe comme affichée

    if len(classes_displayed) == 10:  # Stopper si toutes les classes sont affichées
        break

plt.tight_layout()  # Ajuster les espacements
plt.show()

