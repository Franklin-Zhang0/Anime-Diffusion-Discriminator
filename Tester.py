import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
import numpy as np

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.Pad(padding, fill=0, padding_mode='constant')(image)

MyTransform = transforms.Compose([
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.Resize(512),
    # transforms.RandomCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])

model = torchvision.models.resnet101()
model.fc=nn.Linear(model.fc.in_features, 2)
softmax = nn.Softmax(dim=1)
criteria = nn.CrossEntropyLoss()

try:
    model.load_state_dict(torch.load("./model/model.pth"))
except:
    raise Exception("NO MODEL FILE FOUND IN ./model/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

def get_pil(img):
    img = img.cpu().detach().numpy()[0]
    img = np.transpose(img, (1,2,0))
    img = img * np.array([0.5,0.5,0.5]) + np.array([0.5,0.5,0.5])
    img[img>1] = 1
    img[img<0] = 0
    img = img * 255
    img = img.astype(np.uint8)
    img = transforms.ToPILImage()(img)
    return img

def pred(img):    
    model.eval()
    epochs = 20
    lr = 2e-2
    threshold = 2e-2
    img = MyTransform(img)
    img = img[None,:]
    img = img.to(device)
    img.requires_grad = True
    img_optim = torch.optim.Adam([img], lr=lr)
    # pred = model(img)
    # pred = softmax(pred)
    for i in range(epochs):
        pred = model(img)
        # pred = softmax(pred)
        loss = criteria(pred[0], torch.tensor([0,1],dtype=torch.float32).to(device))
        img_optim.zero_grad()
        loss.backward()
        # img_optim.step()
        # img.data = img.data - lr * img.grad.data# * loss.data
        # img.data[torch.abs(img.grad.data)>threshold] = img.grad.data[torch.abs(img.grad.data)>threshold]/threshold
        img.data = img.data - torch.log((abs(img.grad.data)+threshold)/threshold)

        print("max_grad: {}".format(torch.max(torch.abs(lr * img.grad.data * loss.data))))
        print("loss: {}".format(loss))
        if(i==0):
            #show the probablity of the original img
            prob = softmax(pred).detach()
        print("img probablity: ", softmax(pred),"\n")
    img=get_pil(img)
    return prob[0].cpu().numpy(), img