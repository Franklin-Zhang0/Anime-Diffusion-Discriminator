# %%
import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np
import os
from PIL import ImageFile, Image
import warnings
warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.Pad(padding, fill=0, padding_mode='constant')(image)

MyTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize(512),
    transforms.RandomCrop((512, 512)),
    transforms.RandomRotation(15),
    # SquarePad(),
    # transforms.Resize((512, 512)),
    # transforms.RandomCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])

# %%
datasets = torchvision.datasets.ImageFolder(root="./dataset",transform=MyTransform)
train_length = int(len(datasets) * 0.8)
val_length = int(len(datasets) * 0.15)
Split_Point = [train_length, val_length, len(datasets) - train_length - val_length]
train_set, val_set, test_set = random_split(datasets, Split_Point)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=10)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=10)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=10)

# %%
# Load Resnet
model = torchvision.models.resnet101()
model.fc=nn.Linear(model.fc.in_features, 2)
if os.path.exists("./model/model.pth"):
    model.load_state_dict(torch.load("./model/model.pth"))

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# %%
if __name__== '__main__':
    num_epochs = 50
    patience = 5
    min_loss = 1e10
    max_epoch = 0
    max_acc = 0
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            print('\r[Eposh - %3d][iteration - %3d / %3d ] training loss: %.4E' % (epoch+1, batch_num, len(train_loader), training_loss/(batch_num+1)), end="")
        scheduler.step()
        print()
        print('[Eposh - %3d] training loss: %.3f' % (epoch+1, training_loss / len(train_loader)))

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                val_loss += loss.item()
                correct += (pred.argmax(dim=1) == target).sum().item()
                total += target.size(0)
        print('[Eposh - %3d] validation loss: %.5f' % (epoch+1, val_loss / len(val_loader)))
        print('[Eposh - %3d] validation accuracy: %.3f' % (epoch+1, correct / total))
        
        if max_acc < correct / len(val_loader):
            max_acc = correct / len(val_loader)
            max_epoch = epoch
            torch.save(model.state_dict(), './model/model.pth')
            print("Best so far! Model Saved!")
        
        if epoch - max_epoch > patience:
            print("Early Stopping")
            break
    
    # %%
    test_loss = 0.0
    correct = 0
    total = 0
    print("Testing")
    model.eval()
    with torch.no_grad():
            for batch_num, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                test_loss += loss.item()
                correct += (pred.argmax(dim=1) == target).sum().item()
                total += target.size(0)
    print('test loss: %.5f' % (test_loss / len(test_loader)))
    print('test accuracy: %.3f' % (correct / total))
