import os
import pandas as pd
print("Content root:", os.listdir("/content"))
if os.path.exists("/content/MYDATASET"):
    print("Inside MYDATASET:", os.listdir("/content/MYDATASET"))
if os.path.exists("/content/realvsfakefaces"):
    print("Inside realvsfakefaces:", os.listdir("/content/realvsfakefaces"))
    from google.colab import files
uploaded = files.upload()  

import zipfile

zip_path = "/content/archive.zip"   
extract_to = "/content/MYDATASET/unzipped"
os.makedirs(extract_to, exist_ok=True)  
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("Extraction done. Contents:", os.listdir(extract_to))

import os
print(os.listdir(extract_to))
import pandas as pd

csv_path = "/content/MYDATASET/unzipped/train.csv"
df = pd.read_csv(csv_path)
df.head() 
import zipfile
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
zip_path = "/content/archive.zip"
extract_to = "/content/MYDATASET/unzipped"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction done. Contents:", os.listdir(extract_to))

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # real = 0, fake = 1
        for label_name, label in [('real', 0), ('fake', 1)]:
            folder = os.path.join(img_dir, label_name)
            for fname in os.listdir(folder):
                if fname.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = CustomDataset("/content/MYDATASET/unzipped/rvf10k/train", transform)
valid_dataset = CustomDataset("/content/MYDATASET/unzipped/rvf10k/valid", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

test_loader = valid_loader  # valid as test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
model.eval()   # evaluation mode
train_total = 0
train_correct = 0
with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()

train_accuracy = train_correct / train_total
print(f"Train Accuracy: {train_accuracy*100:.2f}%")
