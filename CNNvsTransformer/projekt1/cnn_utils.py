import torch
from torch import device, nn
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs: int,
        batch_size: int,
        device):
    
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = train_loss / (len(train_dataloader))
        epoch_acc = (correct / total)*100

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / (len(val_dataloader))
        val_acc = (val_correct/val_total)*100
        print(f'[Epoch: {epoch + 1}] Train loss: {epoch_loss:.2f} | Train acc: {epoch_acc:.2f}% | Val loss: {val_loss:.2f} | Val acc: {val_acc:.2f}%')
        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)
        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)

    fig1, ax_acc = plt.subplots()
    plt.plot(train_acc_log)
    plt.plot(val_acc_log)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(train_loss_log)
    plt.plot(val_loss_log)
    plt.show()

def test(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn,
        device):

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = test_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n--- Test Results ---")
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    class_names = dataloader.dataset.classes
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        all_labels, 
        all_preds, 
        display_labels=class_names,
        cmap='Blues', 
        ax=ax,
        colorbar=False,
        xticks_rotation=45
    )
    plt.title('Confusion Matrix')
    plt.show()

class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30752, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fc(x)
        return x
    
class batchCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spłaszcza mapy cech do 1x1, uniezależniając sieć od rozmiaru obrazka
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
class batchdropCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling - rozwiązuje problem "magicznej liczby" (np. 30752) 
        # Niezależnie od rozmiaru wejścia, zredukuje mapy cech do rozmiaru 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Warstwy w pełni połączone 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Główna ścieżka bloku
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Połączenie skrótowe (Shortcut / Residual Connection)
        self.shortcut = nn.Sequential()
        # Jeśli zmieniamy rozmiar (stride > 1) lub liczbę kanałów, shortcut też musi się dostosować
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Zapisujemy wejście
        identity = self.shortcut(x)
        
        # Przepuszczamy przez warstwy
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Dodajemy wejście do wyjścia przed ostatnią aktywacją
        out += identity
        out = self.relu(out)
        
        return out

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Warstwa wejściowa 
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Budujemy sieć z bloków rezydualnych
        self.layer1 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        self.layer2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.layer3 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.prep(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        
        return x