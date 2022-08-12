import cgan_mnist_imbalance_generator
from cgan_mnist_imbalance_generator import *

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset, Subset
import torch.utils.data as data_utils
from tensorboardX import SummaryWriter

train_loader_collection = [trainloader_mnist_1, trainloader_mnist_2, trainloader_mnist_3]
settings = [0, 1, 2] #The different imbalanced settings
batch_size = 16 ##same batch_size used for the Imbalanced Generator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
    
    
generator1 = Generator().cuda()
generator2 = Generator().cuda()
generator3 = Generator().cuda()

generator_collection = [generator1, generator2, generator3]

discriminator1 = Discriminator().cuda()
discriminator2 = Discriminator().cuda()
discriminator3 = Discriminator().cuda()

discriminator_collection = [discriminator1, discriminator2, discriminator3]

criterion = nn.BCELoss()

d_optimizer1 = torch.optim.Adam(discriminator1.parameters(), lr=1e-4)
g_optimizer1 = torch.optim.Adam(generator1.parameters(), lr=1e-4)

d_optimizer2 = torch.optim.Adam(discriminator2.parameters(), lr=1e-4)
g_optimizer2 = torch.optim.Adam(generator2.parameters(), lr=1e-4)

d_optimizer3 = torch.optim.Adam(discriminator3.parameters(), lr=1e-4)
g_optimizer3 = torch.optim.Adam(generator3.parameters(), lr=1e-4)


g_optimizer_collection = [g_optimizer1, g_optimizer2, g_optimizer3]
d_optimizer_collection = [d_optimizer1, d_optimizer2, d_optimizer3]

writer = SummaryWriter()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()

num_epochs = 150

for setting in zip(settings):
    setting = int(''.join(map(str, setting)))
        
    print('-------------------')
    if setting == 0:
        print('SETTING: Half-Split Imbalance')
    elif setting == 1:
        print('SETTING: MultiMajority')
    elif setting == 2:
        print('SETTING: MultiMinority')
    print('-------------------')

    for epoch in range(num_epochs):
        print('Setting: {} \tEpoch: {}'.format(setting, epoch))

        for i, (images, labels) in enumerate(train_loader_collection[setting]):
        
            step = epoch * len(train_loader_collection[setting]) + i + 1
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            generator_collection[setting].train()
        
            d_loss = discriminator_train_step(len(real_images), discriminator_collection[setting],
                                            generator_collection[setting], d_optimizer_collection[setting], criterion,
                                            real_images, labels)
        

            g_loss = generator_train_step(batch_size, discriminator_collection[setting], generator_collection[setting], g_optimizer_collection[setting], criterion)
        
            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)

    print('Finished Setting:', setting)

torch.save(generator1.state_dict(), 'generator1_state.pt')
torch.save(generator2.state_dict(), 'generator2_state.pt')
torch.save(generator3.state_dict(), 'generator3_state.pt')

def image_generator(NUM, digit, setting):
    generator_collection[setting].eval()
    output_container = torch.tensor((), device=torch.device('cuda'))

    for i in range(NUM):
        z = Variable(torch.randn(1, 100)).cuda()
        label = torch.LongTensor([digit]).cuda()
        with torch.no_grad():
            img = generator_collection[setting](z, label)
            img = img.reshape(1, 1, 28, 28)
        
        output_container = torch.cat((output_container, img), 0)

    return output_container


def generate_digit(generator, digit):
    z = Variable(torch.randn(1, 100)).cuda()
    label = torch.LongTensor([digit]).cuda()
    img = generator(z, label).data.cpu()
    image = img.reshape(28, 28)
    fig1, (ax1)= plt.subplots(1, sharex = True, sharey = False)
    ax1.title.set_text(f'CGAN Reconstruction of the Class: {digit}')
    ax1.imshow(image)
    
## Setting One (Half-Split Imbalance)
CGAN_trainset_setting_one = torch.cat((image_generator(4000, 0, 0), image_generator(4000, 1, 0), image_generator(4000, 2, 0), 
                                       image_generator(4000, 3, 0), image_generator(4000, 4, 0)), 0)

CGAN_train_labels_setting_one = torch.cat((torch.tensor([0]*4000), torch.tensor([1]*4000), torch.tensor([2]*4000), torch.tensor([3]*4000), torch.tensor([4]*4000)), 0)
CGAN_setting_one_dataset = TensorDataset(CGAN_trainset_setting_one, CGAN_train_labels_setting_one)

CGAN_S1_MNIST_trainloader = DataLoader(CGAN_setting_one_dataset, batch_size=16, shuffle=True) 

## Setting Two (Multimajority)
CGAN_setting_two_dataset = TensorDataset(image_generator(5742, 9, 1), torch.tensor([9]*5742))
CGAN_S2_MNIST_trainloader = DataLoader(CGAN_setting_two_dataset, batch_size=16, shuffle=True)

## Setting Three (Multiminority)
CGAN_trainset_setting_three = torch.cat((image_generator(5742, 0, 2), image_generator(5742, 1, 2), image_generator(5742, 2, 2), 
                                       image_generator(5742, 3, 2), image_generator(5742, 4, 2), image_generator(5742, 5, 2), 
                                       image_generator(5742, 6, 2), image_generator(5742, 7, 2), image_generator(5742, 8, 2)),  0)

CGAN_train_labels_setting_three = torch.cat((torch.tensor([0]*5742), torch.tensor([1]*5742), torch.tensor([2]*5742), torch.tensor([3]*5742), torch.tensor([4]*5742),
                                             torch.tensor([5]*5742), torch.tensor([6]*5742), torch.tensor([7]*5742), torch.tensor([8]*5742)), 0)

CGAN_setting_three_dataset = TensorDataset(CGAN_trainset_setting_three, CGAN_train_labels_setting_three)

CGAN_S3_MNIST_trainloader = DataLoader(CGAN_setting_three_dataset, batch_size=16, shuffle=True)