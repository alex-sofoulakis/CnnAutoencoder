# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, GaussianBlur
import matplotlib.pyplot as plt
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # Batch, 32, 123
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),  # batch, 64, 61, 61
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),  # Batch, 128, 30
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # batch, 64, 30
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 64, 15
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, blur=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        Xblur = X
        if blur:
             transform = GaussianBlur((5, 9), sigma=(1, 12))
             Xblur = transform(X)


        pred = model(Xblur)
        loss = loss_fn(pred, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            plot_image(X[2], Xblur[2], pred[2])
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:.7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, blur=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            Xblur = X
            if blur:
                transform = GaussianBlur((5, 9), sigma=(1, 12))  # comment out for plain run
                Xblur = transform(X)

              # uncomment for plain run

            pred = model(Xblur)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    print(f"Avg loss in Testing: {test_loss:.7f}\n")

def plot_image_set(image_set):
    fig, axes = plt.subplots(1, len(image_set))
    for ax, image in zip(axes, image_set):
        ax.imshow(image)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
    plt.show()

def convert_to_numpy(image_tensor):
    image = image_tensor.to('cpu')
    return torch.permute(image, (1, 2, 0)).detach().numpy()

def plot_image(X, Xblur, pred):
    X, Xblur, pred = convert_to_numpy(X), convert_to_numpy(Xblur), convert_to_numpy(pred)
    plot_image_set([X, Xblur, pred])

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transforms = Compose([
        Resize((256, 256)),
        ToTensor()
    ])
    train_data = datasets.ImageFolder(root='/tempData/tempData/train', transform=transforms)
    test_data = datasets.ImageFolder(root='/tempData/tempData/test', transform=transforms)

    batch_size = 14
    learning_rate = 1 * 1e-3
    epochs = 45

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    torch.cuda.empty_cache()
    start_time = time.time()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        start = time.time()
        train_loop(train_data_loader, model, loss_fn, optimizer,blur=True)
        test_loop(test_data_loader, model, loss_fn,blur=True)
        end = time.time()
        print("Duration of epoch is: %f \n" % (end - start))

    end_time = time.time()
    print("lr = " + str(learning_rate) + " batch_size = " + str(batch_size))
    print("Total time taken: %.2f seconds" % (end_time - start_time))
