# -*- coding: utf-8 -*-


import torch
from torch import nn, flatten
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose, GaussianBlur, ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.metrics import mean_squared_error
from PIL import Image
import time


    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1), #Batch, 32, 123
            nn.ReLU(),
            nn.Conv2d(32, 64,3, stride = 2), #batch, 64, 61,61
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, stride = 2), #Batch, 128, 30
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1), #batch, 64, 30
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride = 2, padding = 1), #64, 15
            
         )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride = 2, padding = 1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 3, stride = 1, padding = 1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride = 2, output_padding=1),
            nn.Sigmoid()
            )
       
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer): #this is our training loop
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        transform = GaussianBlur((5,9), sigma = (1, 12))
        X = X.to(device)
        Xgray = transform(X) #transform to blurry, comment out for plain image reconstruction
       
        #Xgray = X #uncomment for plain image reconstruction
        
       
        
        # Compute prediction and loss
           
        pred = model(Xgray) #feed input to the network
        
        loss = loss_fn(pred, X) #calculate loss

        # Backpropagation
        optimizer.zero_grad() #delete previous gradients
        loss.backward() #calculate new gradients (backwards propagation)
        optimizer.step() #update weights accordingly
        if batch % 100 == 0:
            plotImage(X[2], Xgray[2], pred[2])
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          
           
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad(): #as to not affect the gradients, similar to .detach
        for X, y in dataloader:
            X = X.to(device)
            transform = GaussianBlur((5,9), sigma = (1, 12)) #comment out for plain run

            Xgray = transform(X)
           # Xgray = X  #uncomment for plain run
           

            pred = model(Xgray)
            test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches
    print(f"Avg loss in Testing : {test_loss:>8f} \n")
    
def plotImage(X, Xgray, pred): #plot images in sets of 3 (normal, transformed and reconstructed)
     X = X.to('cpu')
     X = torch.permute(X,(1,2,0))
     X = X.detach().numpy()
     
     Xgray = Xgray.to('cpu')
     Xgray = torch.permute(Xgray,(1,2,0))
     Xgray = Xgray.detach().numpy()
     
     pred = pred.to('cpu')
     pred = torch.permute(pred,(1,2,0))
     pred = pred.detach().numpy()
     
     plt.subplot(1,3,1)
     plt.gca().axes.get_yaxis().set_visible(False)
     plt.gca().axes.get_xaxis().set_visible(False)
     plt.imshow(X)
     
     plt.subplot(1,3,2)
     plt.gca().axes.get_yaxis().set_visible(False)
     plt.gca().axes.get_xaxis().set_visible(False)
     plt.imshow(Xgray)
     
     plt.subplot(1,3,3)
     plt.imshow(pred)
     plt.gca().axes.get_yaxis().set_visible(False)
     plt.gca().axes.get_xaxis().set_visible(False)

     
     plt.show()
    
     
    


device = "cuda:0" if torch.cuda.is_available() else "cpu" #get available device

transforms = Compose([
    Resize((256,256)),
    ToTensor()
    ]) 
train_data = datasets.ImageFolder(root='tempData/train', transform = transforms) #folder name is tempData
test_data = datasets.ImageFolder(root='tempData/test', transform = transforms) #folder name is tempData
 

batch_size = 14
learning_rate = 1*1e-3

epochs = 35

train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size,shuffle=True)


model = Autoencoder().to(device) #initialize the network
loss = nn.MSELoss() #initialize loss function
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)


torch.cuda.empty_cache()
startT = time.time()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start = time.time()
    train_loop(train_data_loader, model, loss, optimizer) #start epoch's training
    test_loop(test_dataloader, model, loss) #start epoch's testing
    end = time.time()
    print("Duration of epoch is: %f \n"  %(end-start))
endT = time.time()


print("lr = "+ str(learning_rate) + " batch_size = " +str(batch_size))



