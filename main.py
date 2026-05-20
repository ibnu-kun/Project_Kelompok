import os
import cv2
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from skimage.feature import hog

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. DOWNLOAD & LOAD DATASET
# ==========================================
print("Mulai mengunduh dataset dari Kaggle...")
path = kagglehub.dataset_download("rhythmghai/ai-vs-real-images-dataset")
print("Path dataset:", path)

IMG_SIZE = 64 # Ukuran gambar diseragamkan ke 64x64
image_data = []
labels = []
class_names = []

print("Membaca dan memproses gambar...")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            label_name = os.path.basename(root)
            
            if label_name not in class_names:
                class_names.append(label_name)
                
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                image_data.append(img_resized)
                labels.append(class_names.index(label_name))

image_data = np.array(image_data)
labels = np.array(labels)

print(f"Total gambar berhasil dimuat: {len(image_data)}")
print(f"Kelas terdeteksi: {class_names}")

if len(image_data) < 500:
    print("Peringatan: Jumlah dataset kurang dari 500 gambar sesuai aturan tugas!")

# ==========================================
# 2. SPLIT DATA (DATA LATIH & DATA UJI)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
print(f"Data Latih: {len(X_train)} gambar, Data Uji: {len(X_test)} gambar")

# ==========================================
# 3. METODE 1: KLASIK (HOG + SVM)
# ==========================================
print("\n--- Memulai Pelatihan Metode Klasik: SVM + HOG ---")

def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

print("Mengekstrak fitur HOG...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("Melatih model SVM...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_hog, y_train)

print("Evaluasi SVM:")
svm_predictions = svm_model.predict(X_test_hog)
print(classification_report(y_test, svm_predictions, target_names=class_names))


# ==========================================
# 4. METODE 2: MODERN (CNN DASAR DENGAN PYTORCH)
# ==========================================
print("\n--- Memulai Pelatihan Metode Modern: CNN Dasar (PyTorch) ---")

# Preprocessing: Transpose dimensi ke format PyTorch (Channel, Height, Width)
X_train_pt = np.transpose(X_train.astype('float32') / 255.0, (0, 3, 1, 2))
X_test_pt = np.transpose(X_test.astype('float32') / 255.0, (0, 3, 1, 2))

train_dataset = TensorDataset(torch.tensor(X_train_pt), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test_pt), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Arsitektur CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dictionary untuk menyimpan history training mirip Keras/TensorFlow
history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

print("Melatih model CNN (5 Epochs)...")
for epoch in range(5):
    # --- PHASE TRAINING ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total_train += targets.size(0)
        correct_train += (preds == targets).sum().item()
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # --- PHASE VALIDATION (EVALUASI DATA UJI) ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (preds == targets).sum().item()
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    # Menyimpan ke history
    history['loss'].append(epoch_train_loss)
    history['accuracy'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_acc)
    
    print(f"Epoch {epoch+1}/5 -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

print("\nEvaluasi Akhir CNN:")
cnn_predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        cnn_predictions.extend(preds.numpy())
cnn_predictions = np.array(cnn_predictions)

print(classification_report(y_test, cnn_predictions, target_names=class_names))


# ==========================================
# 5. VISUALISASI GRAPHICS & CHART (DIAGRAM PERFORMA)
# ==========================================
print("\n--- Menampilkan Visualisasi Performa ---")

# 1. Grafik Akurasi dan Loss Pelatihan CNN (Line Charts)
plt.figure(figsize=(12, 5))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(range(1, 6), history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Akurasi Pelatihan CNN (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), history['loss'], label='Training Loss', marker='o')
plt.plot(range(1, 6), history['val_loss'], label='Validation Loss', marker='s')
plt.title('Loss Pelatihan CNN (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Visualisasi Confusion Matrix berdampingan menggunakan Seaborn Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix SVM (Metode Klasik) - Tema Biru
cm_svm = confusion_matrix(y_test, svm_predictions)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('Confusion Matrix: SVM (Metode Klasik)')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Aktual')

# Confusion Matrix CNN (Metode Modern) - Tema Hijau
cm_cnn = confusion_matrix(y_test, cnn_predictions)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Confusion Matrix: CNN (Metode Modern)')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')

plt.tight_layout()
plt.show()

print("\n==========================================")
print("PROSES SELESAI. Silakan tangkap layar (screenshot) grafik di atas untuk isi Makalah IEEE kamu.")
print("==========================================")