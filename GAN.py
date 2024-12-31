import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and preprocess MNIST data
from torchvision import datasets, transforms

# MNIST preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
])
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=True)

# Parameters
latent_dim = 100
img_size = 28 * 28  # Flattened image size
batch_size = 32
epochs = 30000
sample_interval = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Initialize generator and discriminator
generator = Generator(latent_dim, img_size).to(device)
discriminator = Discriminator(img_size).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Function to save generated images
def save_images(epoch):
    z = torch.randn(25, latent_dim).to(device)
    gen_imgs = generator(z).view(-1, 1, 28, 28)
    gen_imgs = (gen_imgs + 1) / 2.0  # Rescale to [0, 1]

    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[i * 5 + j].squeeze().cpu().detach().numpy(), cmap='gray')
            axs[i, j].axis('off')
    fig.savefig(f"gan_images/epoch_{epoch}.png")
    plt.close()

# Create output directory
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# Training loop
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # Configure real and fake labels
        real = torch.ones((imgs.size(0), 1)).to(device)
        fake = torch.zeros((imgs.size(0), 1)).to(device)

        # Real images
        real_imgs = imgs.view(imgs.size(0), -1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Logging
        if i % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Batch {i}/{len(data_loader)}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save images at intervals
    if epoch % sample_interval == 0:
        save_images(epoch)
