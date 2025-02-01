import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def save_weights_biases(model):
"""Pour enregistrer les poids et biais dans un fichier texte"""    

    conv1_weights = model.layer1[0].weight.detach().numpy().reshape(-1)  # Flattening the 4D weights to 1D
    conv1_biases = model.layer1[0].bias.detach().numpy()  # Biases are already 1D
    
    conv2_weights = model.layer2[0].weight.detach().numpy().reshape(-1)  # Flattening the 4D weights to 1D
    conv2_biases = model.layer2[0].bias.detach().numpy()  # Biases are already 1D
    
    fc1_weights = model.fc1.weight.detach().numpy()  
    fc1_biases = model.fc1.bias.detach().numpy() 
    
    fc2_weights = model.fc2.weight.detach().numpy()  
    fc2_biases = model.fc2.bias.detach().numpy()  
    
    # enregistrer dans les fichiers texte
    np.savetxt('conv1_weights.txt', conv1_weights) 
    np.savetxt('conv1_biases.txt', conv1_biases)
    np.savetxt('conv2_weights.txt', conv2_weights)
    np.savetxt('conv2_biases.txt', conv2_biases)
    np.savetxt('fc1_weights.txt', fc1_weights)
    np.savetxt('fc1_biases.txt', fc1_biases)
    np.savetxt('fc2_weights.txt', fc2_weights)
    np.savetxt('fc2_biases.txt', fc2_biases)



# Hyperparamètres
batch_size = 15
num_epochs = 18
learning_rate = 0.001

# Définition des transformations pour les images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Redimensionner les images à 28x28
    transforms.ToTensor(),       # Convertir les images en tenseurs PyTorch
    #transforms.Normalize((0.5,), (0.5,))  # Normalisation (moyenne=0.5, écart-type=0.5)
])

# Charger les données
train_dir = "chiffre/training"  
test_dir = "chiffre/testing"    

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Créer les DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Afficher les classes chargées
print(f"Classes détectées : {train_dataset.classes}")

# Définition du modèle
class CNN(nn.Module):
    def __init__(self, num_classes): 
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Instancier le modèle, la fonction de perte et l'optimiseur
num_classes = 10
model = CNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #model.parameters()

# Fonction d'entraînement
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return train_losses

# Fonction de validation
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            print(images.shape)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

# Entraîner le modèle
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs)

# Évaluer le modèle
test_accuracy = evaluate_model(model, test_loader)

save_weights_biases(model)

# Visualiser la courbe de perte
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()





