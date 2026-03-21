import torch
from torch import device, nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import math



class Scheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.max_lr = max_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1  
        if self.current_step <= self.warmup_steps: # warmup
            # liniowo zwiekszamy lr
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else: # po warmup
            # liniowo zmniejszamy LR
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.max_lr * (1-progress)
        self.optimizer.param_groups[0]['lr'] = lr

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'max_lr': self.max_lr
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.max_lr = state_dict['max_lr']


        
def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs: int,
        batch_size: int,
        patience = 5,
        device = 'cuda',
        save_dir = 'transformer_checkpoints',
        model_name = 'vit_model',
        resume = False,
        scheduler = None,
        gradient_clipping = 0):

    # do zapisywania
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_val_acc = 0
    
    if resume:
        checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pth')
        if os.path.exists(checkpoint_path):
            print(f"Wznawiam trening z {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0)
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Kontynuuję od epoki {start_epoch}")
        else:
            print("Brak checkpointu, zaczynam od początku")
    
    # do wykresow
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    # plik csv do zapisywania
    csv_path = os.path.join(save_dir, f'{model_name}_training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'timestamp'])

    # do najlepszego modelu
    best_epoch = 0

    # do early stopping
    patience = patience
    not_improving = 0

    for epoch in range(start_epoch, n_epochs):

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # progress bar
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_acc = (correct / total)*100
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%', 'lr': f'{current_lr:.2e}'})

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

        # zapisujemy do pliku csv
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f'{epoch_loss:.4f}', f'{epoch_acc:.2f}', 
                           f'{val_loss:.4f}', f'{val_acc:.2f}', f'{current_lr:.6f}', 
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        # checkpointy co 5 epok
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_path = os.path.join(save_dir, f'{model_name}_checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'\nZapisano checkpoint: {checkpoint_path}\n')

        # zapisywanie najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            not_improving = 0
            
            best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, best_model_path)
            print(f'\nNowy najlepszy model: Val acc: {val_acc:.2f}% (epoch {epoch+1})\n')
        else:
            not_improving += 1

        # early stopping
        if not_improving > patience:
            print(f'\nEarly stopping po {epoch+1} epokach (patience: {patience})')
            break

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
        device,
        save_dir = 'transformer_checkpoints',
        model_name = 'vit_model'):

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # znowu progress bar
        for inputs, labels in tqdm(dataloader, desc='Testowanie'):
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

    # zapisywanie wynikow
    results_path = os.path.join(save_dir, f'{model_name}_test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"{'='*50}\n")
        f.write(f"WYNIKI TESTU - {model_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Loss:      {avg_loss:.4f}\n")
        f.write(f"Accuracy:  {acc*100:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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



def load_checkpoint(model, optimizer, checkpoint_path, device, scheduler=None):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0)

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Wczytano checkpoint z epoki {epoch}")
    print(f"\nVal acc: {checkpoint.get('val_acc', 'N/A')}%")
    
    return model, optimizer, epoch, best_val_acc
