#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split


# In[ ]:


val_size = 0.2
batch_size = 32
device = torch.device('cuda')


# In[ ]:


transform_CIFAR_10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class_names_CIFAR10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
nb_classes_CIFAR10 = len(class_names_CIFAR10)


# In[ ]:


def get_labels_and_class_counts(labels_list):
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts


# In[ ]:


class ImbalanceGeneratorCIFAR10(Dataset):
    def __init__(self, num_samples, root, train, download, transform):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform_CIFAR_10)
        self.train = train
        self.num_samples = num_samples
        self.idxs = self.resample()

    def get_labels_and_class_counts(self):
        return self.labels, self.imbal_class_counts

    def resample(self):
        if self.train:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.targets)
        else:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.targets)

        class_indices = [np.where(targets == i)[0] for i in range(nb_classes_CIFAR10)]

        self.imbal_class_counts = [
            int(prop)
            for count, prop in zip(class_counts, self.num_samples)
        ]

        idxs = []
        for c in range(nb_classes_CIFAR10):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        img, target = self.dataset[self.idxs[index]]
        return img, target

    def __len__(self):
        return len(self.idxs)


# In[ ]:


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dataset_creator(seed_number, num_samples_train, num_samples_test):
    g = torch.Generator()
    g.manual_seed(seed_number)

    trainset = ImbalanceGeneratorCIFAR10(num_samples_train, root='/home/anishg/datasets/CIFAR-10', train=True, download=True, transform=transform_CIFAR_10)
    testset = ImbalanceGeneratorCIFAR10(num_samples_test, root='/home/anishg/datasets/CIFAR-10', train=False, download=True, transform=transform_CIFAR_10)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, pin_memory=device)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, pin_memory=device)

    return trainloader, testloader


# In[ ]:


##Half Split Imbalance
setting1_cifar10 = dataset_creator(4, np.hstack(([1000]*5, [5000]*5)), np.hstack(([200] * 5, [1000]*5)))
trainloader_cifar10_1 = setting1_cifar10[0]
testloader_cifar10_1 = setting1_cifar10[1]

##Multimajority
setting2_cifar10 = dataset_creator(5, np.hstack(([5000]*9, [50])), np.hstack(([1000] * 9, [10])))
trainloader_cifar10_2 = setting2_cifar10[0]
testloader_cifar10_2 = setting2_cifar10[1]

##Multiminority
setting3_cifar10 = dataset_creator(6, np.hstack(([50]*9, [5000])), np.hstack(([10] * 9, [1000])))
trainloader_cifar10_3 = setting3_cifar10[0]
testloader_cifar10_3 = setting3_cifar10[1]


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from torch.utils.data import TensorDataset, DataLoader


# In[ ]:


def sampler_CIFAR10(setting, oversample, undersample, smote, seed):
    if setting == 'Half-Split':
        sample_size = 30000
        ros_sample_size = 50000
        rus_sample_size = 10000
        trainloader = trainloader_cifar10_1

    if setting == 'Multimajority':
        sample_size = 45050
        ros_sample_size = 50000
        rus_sample_size = 500
        trainloader = trainloader_cifar10_2

    if setting == 'Multiminority':
        sample_size = 5450
        ros_sample_size = 50000
        rus_sample_size = 500
        trainloader = trainloader_cifar10_3

    output_container_data = torch.tensor((), device=device)
    output_container_labels = torch.tensor((), device=device)

    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        output_container_data = torch.cat((output_container_data, data), 0)
        output_container_labels = torch.cat((output_container_labels, labels), 0)

    output_container_data_numpy = output_container_data.cpu().detach().numpy()
    output_container_data_numpy_reshaped = output_container_data_numpy.reshape(sample_size, 3*32*32)
    output_container_labels_numpy = output_container_labels.cpu().detach().numpy()

    if oversample==True:
        ros = RandomOverSampler(random_state=seed)
        sampled_data, sampled_labels = ros.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(ros_sample_size, 3, 32, 32)
    
    if undersample==True:
        rus = RandomUnderSampler(random_state=seed)
        sampled_data, sampled_labels = rus.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(rus_sample_size, 3, 32, 32)

    if smote==True:
        smote = SMOTE(random_state=seed)
        sampled_data, sampled_labels = smote.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(ros_sample_size, 3, 32, 32)

    sampled_data = torch.from_numpy(sampled_data)
    sampled_labels = torch.from_numpy(sampled_labels)
    sampled_labels = sampled_labels.long()

    Resampled_Dataset = TensorDataset(sampled_data, sampled_labels)
    trainloader_new = DataLoader(Resampled_Dataset, batch_size=16, shuffle=True) 

    return trainloader_new


# In[ ]:


##Undersample Datasets for each Setting:
CIFAR10_S1_Undersample_Trainloader = sampler_CIFAR10('Half-Split', oversample=False, undersample=True, smote=False, seed=1)
CIFAR10_S2_Undersample_Trainloader = sampler_CIFAR10('Multimajority', oversample=False, undersample=True, smote=False, seed=2)
CIFAR10_S3_Undersample_Trainloader = sampler_CIFAR10('Multiminority', oversample=False, undersample=True, smote=False, seed=3)

##Oversample Datasets for each Setting:
CIFAR10_S1_Oversample_Trainloader = sampler_CIFAR10('Half-Split', oversample=True, undersample=False, smote=False, seed=1)
CIFAR10_S2_Oversample_Trainloader = sampler_CIFAR10('Multimajority', oversample=True, undersample=False, smote=False, seed=2)
CIFAR10_S3_Oversample_Trainloader = sampler_CIFAR10('Multiminority', oversample=True, undersample=False, smote=False, seed=3)

##SMOTE Datasets for each Setting:
CIFAR10_S1_SMOTE_Trainloader = sampler_CIFAR10('Half-Split', oversample=False, undersample=False, smote=True, seed=1)
CIFAR10_S2_SMOTE_Trainloader = sampler_CIFAR10('Multimajority', oversample=False, undersample=False, smote=True, seed=2)
CIFAR10_S3_SMOTE_Trainloader = sampler_CIFAR10('Multiminority', oversample=False, undersample=False, smote=True, seed=3)


# In[ ]:


import torch
from torch.nn import Module
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import models

import random
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from imblearn.metrics import (geometric_mean_score, sensitivity_score, 
                              specificity_score)
from sklearn.metrics import (balanced_accuracy_score, precision_score, 
                             recall_score, f1_score)


import scipy as sc
import matplotlib.style

params = {'legend.fontsize': 14,
          'axes.labelsize': 14,
          'axes.titlesize': 14,
          'xtick.labelsize' :14,
          'ytick.labelsize': 13,
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 0.8,
          'mathtext.fontset' : 'stix',
          'mathtext.rm'      : 'serif',
          'font.family'      : 'serif',
          'font.serif'       : "Times New Roman", # or "Times"          
         }
matplotlib.rcParams.update(params)


# In[ ]:


if torch.cuda.is_available():
    print("life is good")

device = torch.device("cuda")


# In[ ]:


trainloader_collection = [
                          [trainloader_cifar10_1, trainloader_cifar10_2, trainloader_cifar10_3],
                           [CIFAR10_S1_Undersample_Trainloader, CIFAR10_S2_Undersample_Trainloader, CIFAR10_S3_Undersample_Trainloader],
                           [CIFAR10_S1_Oversample_Trainloader, CIFAR10_S2_Oversample_Trainloader, CIFAR10_S3_Oversample_Trainloader],
                           [CIFAR10_S1_SMOTE_Trainloader, CIFAR10_S2_SMOTE_Trainloader, CIFAR10_S3_SMOTE_Trainloader]
                         ]

testloader_collection = [testloader_cifar10_1, testloader_cifar10_2, testloader_cifar10_3]

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
output_size_network = len(classes)

epochs=150


# In[ ]:


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
  
model1 = ResNet18()
model1 = nn.DataParallel(model1)
model1 = model1.to(device)

model2 = ResNet18()
model2 = nn.DataParallel(model2)
model2 = model2.to(device)

model3 = ResNet18()
model3 = nn.DataParallel(model3)
model3 = model3.to(device)

model4 = ResNet18()
model4 = nn.DataParallel(model4)
model4 = model4.to(device)

model5 = ResNet18()
model5 = nn.DataParallel(model5)
model5 = model5.to(device)

model6 = ResNet18()
model6 = nn.DataParallel(model6)
model6 = model6.to(device)

model7 = ResNet18()
model7 = nn.DataParallel(model7)
model7 = model7.to(device)

model8 = ResNet18()
model8 = nn.DataParallel(model8)
model8 = model8.to(device)

model9 = ResNet18()
model9 = nn.DataParallel(model9)
model9 = model9.to(device)

model10 = ResNet18()
model10 = nn.DataParallel(model10)
model10 = model10.to(device)

model11 = ResNet18()
model11 = nn.DataParallel(model11)
model11 = model11.to(device)

model12 = ResNet18()
model12 = nn.DataParallel(model12)
model12 = model12.to(device)


# In[ ]:


model = [[model1, model2, model3], [model4, model5, model6], [model7, model8, model9], [model10, model11, model12]]


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-5)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-5)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001, weight_decay=1e-5)
optimizer4 = optim.Adam(model4.parameters(), lr=0.001, weight_decay=1e-5)
optimizer5 = optim.Adam(model5.parameters(), lr=0.001, weight_decay=1e-5)
optimizer6 = optim.Adam(model6.parameters(), lr=0.001, weight_decay=1e-5)
optimizer7 = optim.Adam(model7.parameters(), lr=0.001, weight_decay=1e-5)
optimizer8 = optim.Adam(model8.parameters(), lr=0.001, weight_decay=1e-5)
optimizer9 = optim.Adam(model9.parameters(), lr=0.001, weight_decay=1e-5)
optimizer10 = optim.Adam(model10.parameters(), lr=0.001, weight_decay=1e-5)
optimizer11 = optim.Adam(model11.parameters(), lr=0.001, weight_decay=1e-5)
optimizer12 = optim.Adam(model12.parameters(), lr=0.001, weight_decay=1e-5)

optimizer = [[optimizer1, optimizer2, optimizer3], [optimizer4, optimizer5, optimizer6], [optimizer7, optimizer8, optimizer9], [optimizer10, optimizer11, optimizer12]]

sampling_methods = [0, 1, 2, 3]
settings = [0, 1, 2]


# In[ ]:


train_loss_hist = []

for sampling_method in zip(sampling_methods):
    sampling_method = int(''.join(map(str, sampling_method)))

    for setting in zip(settings): 
        setting = int(''.join(map(str, setting)))
    
        print('-------------------')
        if sampling_method == 0:
            print('SAMPLER: None')
        elif sampling_method == 1:
            print('SAMPLER: RUS')
        elif sampling_method == 2:
            print('SAMPLER: ROS')
        elif sampling_method == 3:
            print('SAMPLER: SMOTE')

        if setting == 0:
            print('Half-Split Imbalance')
        elif setting == 1:
            print('Multimajority')
        elif setting == 2:
            print('Multiminority')
        print('-------------------')

        np.random.seed(setting)
        torch.manual_seed(setting)
        random.seed(setting)
        torch.cuda.manual_seed(setting)

        for epoch in range(1, epochs+1):  
            train_loss = 0.0

            model[sampling_method][setting].train()
            for data, labels in trainloader_collection[sampling_method][setting]:
                data, labels = data.to(device), labels.to(device)
                optimizer[sampling_method][setting].zero_grad()
                outputs = model[sampling_method][setting](data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer[sampling_method][setting].step()

                train_loss += loss.item()

            train_loss = train_loss/len(trainloader_collection[sampling_method][setting])
            train_loss_hist.append(train_loss)

            print('Setting: {} \tEpoch: {} \tTraining Loss: {:.3f}'.format(setting, 
                epoch, train_loss))
    
        if setting == 0:
            print('Finished Training for setting', setting)
        elif setting == 1:
            print('Finished Training for setting', setting)
        elif setting == 2:
            print('Finished Training for setting', setting) 


# In[ ]:


model_updated = [[model1, model2, model3], [model4, model5, model6], [model7, model8, model9], [model10, model11, model12]]


# In[ ]:


global_accuracy = []
test_loss_hist = []
labels_list = []
pred_list= []
class_accuracy_model =[]

for sampling_method in zip(sampling_methods):
    sampling_method = int(''.join(map(str, sampling_method)))

    for setting in zip(settings): 
        setting = int(''.join(map(str, setting)))
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        
        model_updated[sampling_method][setting].eval()
        
        with torch.no_grad():
            for data, labels in testloader_collection[setting]:
                images, labels = data.to(device), labels.to(device)
                output = model_updated[sampling_method][setting](images)
    
                loss = criterion(output, labels)
                test_loss += loss.item()

                _, pred = torch.max(output, 1)    

                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
                for label, p in zip(labels, pred):
                    if label == p:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

                pred = pred.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                pred_list.append(pred)
                labels_list.append(labels)

        test_loss = test_loss/len(testloader_collection[setting])
        test_loss_hist.append(test_loss)
        
        accuracy = 100 * correct / total
        global_accuracy.append(accuracy)
        
        for classname, correct_count in correct_pred.items():
            model_seed_accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accuracy_model.append(model_seed_accuracy)


# In[ ]:


print('-------------')
print('SAMPLER: NONE')
print('-------------')
print('Test Loss S1: {:.3f}'.format(test_loss_hist[0]))
print('Test Loss S2: {:.3f}'.format(test_loss_hist[1]))
print('Test Loss S3: {:.3f}'.format(test_loss_hist[2]))

print('-------------')
print('SAMPLER: RUS')
print('-------------')
print('Test Loss S1: {:.3f}'.format(test_loss_hist[3]))
print('Test Loss S2: {:.3f}'.format(test_loss_hist[4]))
print('Test Loss S3: {:.3f}'.format(test_loss_hist[5]))


print('-------------')
print('SAMPLER: ROS')
print('-------------')
print('Test Loss S1: {:.3f}'.format(test_loss_hist[6]))
print('Test Loss S2: {:.3f}'.format(test_loss_hist[7]))
print('Test Loss S3: {:.3f}'.format(test_loss_hist[8]))          


print('-------------')
print('SAMPLER: SMOTE')
print('-------------')
print('Test Loss S1: {:.3f}'.format(test_loss_hist[9]))
print('Test Loss S2: {:.3f}'.format(test_loss_hist[10]))
print('Test Loss S3: {:.3f}'.format(test_loss_hist[11]))


# In[ ]:


print('-------------')
print('SAMPLER: NONE')
print('-------------')
print('Test Accuracy S1: {:.3f}%'.format(global_accuracy[0]))
print('Test Accuracy S2: {:.3f}%'.format(global_accuracy[1]))
print('Test Accuracy S3: {:.3f}%'.format(global_accuracy[2]))

print('-------------')
print('SAMPLER: RUS')
print('-------------')
print('Test Accuracy S1: {:.3f}%'.format(global_accuracy[3]))
print('Test Accuracy S2: {:.3f}%'.format(global_accuracy[4]))
print('Test Accuracy S3: {:.3f}%'.format(global_accuracy[5]))


print('-------------')
print('SAMPLER: ROS')
print('-------------')
print('Test Accuracy S1: {:.3f}%'.format(global_accuracy[6]))
print('Test Accuracy S2: {:.3f}%'.format(global_accuracy[7]))
print('Test Accuracy S3: {:.3f}%'.format(global_accuracy[8]))         


print('-------------')
print('SAMPLER: SMOTE')
print('-------------')
print('Test Accuracy S1: {:.3f}%'.format(global_accuracy[9]))
print('Test Accuracy S2: {:.3f}%'.format(global_accuracy[10]))
print('Test Accuracy S3: {:.3f}%'.format(global_accuracy[11]))


# In[ ]:


class_accuracy_model = np.array_split(class_accuracy_model, 12)
class_accuracy_model


# In[ ]:


slices = [6000, 9010, 1090, 6000, 9010, 1090, 6000, 9010, 1090, 6000, 9010, 1090]

def imbalance_slicer(mylist):
    return [mylist[sum(slices[:i]):sum(slices[:i+1])] for i in range(len(slices))]


# In[ ]:


pred_list = [item for sublist in pred_list for item in sublist]
labels_list = [item for sublist in labels_list for item in sublist]

pred_model1 = imbalance_slicer(pred_list)[0]
pred_model2 = imbalance_slicer(pred_list)[1]
pred_model3 = imbalance_slicer(pred_list)[2]

pred_model4 = imbalance_slicer(pred_list)[3]
pred_model5 = imbalance_slicer(pred_list)[4]
pred_model6 = imbalance_slicer(pred_list)[5]

pred_model7 = imbalance_slicer(pred_list)[6]
pred_model8 = imbalance_slicer(pred_list)[7]
pred_model9 = imbalance_slicer(pred_list)[8]

pred_model10 = imbalance_slicer(pred_list)[9]
pred_model11 = imbalance_slicer(pred_list)[10]
pred_model12 = imbalance_slicer(pred_list)[11]

labels_s1 = imbalance_slicer(labels_list)[0]
labels_s2 = imbalance_slicer(labels_list)[1]
labels_s3 = imbalance_slicer(labels_list)[2]

labels_s4 = imbalance_slicer(labels_list)[3]
labels_s5 = imbalance_slicer(labels_list)[4]
labels_s6 = imbalance_slicer(labels_list)[5]

labels_s7 = imbalance_slicer(labels_list)[6]
labels_s8 = imbalance_slicer(labels_list)[7]
labels_s9 = imbalance_slicer(labels_list)[8]

labels_s10 = imbalance_slicer(labels_list)[9]
labels_s11 = imbalance_slicer(labels_list)[10]
labels_s12 = imbalance_slicer(labels_list)[11]


# In[ ]:


pred_list = [[pred_model1, pred_model2, pred_model3], [pred_model4, pred_model5, pred_model6], 
             [pred_model7, pred_model8, pred_model9], [pred_model10, pred_model11, pred_model12]]

labels_list = [[labels_s1, labels_s2, labels_s3], [labels_s4, labels_s5, labels_s6],
               [labels_s7, labels_s8, labels_s9], [labels_s10, labels_s11, labels_s12]]

f1_micro_list, f1_macro_list = [], []
gmean_micro_list, gmean_macro_list = [], []
bac_list, bac_adj_list = [], []
sens_micro_list, sens_macro_list = [], []
spec_micro_list, spec_macro_list = [], []
prec_micro_list, prec_macro_list = [], []
rec_micro_list, rec_macro_list = [], []

for sampling_method in zip(sampling_methods):
    sampling_method = int(''.join(map(str, sampling_method)))

    for setting in zip(settings):
        setting = int(''.join(map(str, setting)))
        
        f1_micro = f1_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        f1_macro = f1_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)
        
        gmean_micro = geometric_mean_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        gmean_macro = geometric_mean_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        gmean_micro_list.append(gmean_micro)
        gmean_macro_list.append(gmean_macro)
        
        bac = balanced_accuracy_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting])
        bac_adj = balanced_accuracy_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], adjusted=True)
        bac_list.append(bac)
        bac_adj_list.append(bac_adj)
        
        sens_micro = sensitivity_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        sens_macro = sensitivity_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        sens_micro_list.append(sens_micro)
        sens_macro_list.append(sens_macro)
        
        spec_micro = specificity_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        spec_macro = specificity_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        spec_micro_list.append(sens_micro)
        spec_macro_list.append(sens_macro)
                                       
        prec_micro = precision_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        prec_macro = precision_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        prec_micro_list.append(prec_micro)
        prec_macro_list.append(prec_macro)
                                       
        rec_micro = recall_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='micro')
        rec_macro = recall_score(labels_list[sampling_method][setting], pred_list[sampling_method][setting], average='macro')
        rec_micro_list.append(rec_micro)
        rec_macro_list.append(rec_macro)


# In[ ]:


metrics_list = [f1_micro_list, f1_macro_list, gmean_micro_list, gmean_macro_list, bac_list, bac_adj_list, sens_micro_list, 
                sens_macro_list, spec_micro_list, spec_macro_list,
                prec_micro_list, prec_macro_list, rec_micro_list, rec_macro_list]

names = ["F1 Micro", "F1 Macro", "GMean Micro", "GMean Macro", "Balanced Accuracy", "Adjusted Balanced Accuracy", 
         "Sensitivity Micro", "Sensitivity Macro", "Specificity Micro", 
         "Specificity Macro", "Precision Micro", "Precision Macro", "Recall Micro", "Recall Macro"]

for metric, name in zip(metrics_list, names):
    print('---------------')
    print('SAMPLER: NONE')
    print('SETTING: S1')
    print('METRIC:', name)
    print('---------------')
    print(' {:.2f}'.format(metric[0]))
    
    print('---------------')
    print('SAMPLER: NONE')
    print('SETTING: S2')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[1]))
    
    print('---------------')
    print('SAMPLER: NONE')
    print('SETTING: S3')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[2]))
    
    print('\n')

    print('---------------')
    print('SAMPLER: RUS')
    print('SETTING: S1')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[3]))

    print('---------------')
    print('SAMPLER: RUS')
    print('SETTING: S2')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[4]))

    print('---------------')
    print('SAMPLER: RUS')
    print('SETTING: S3')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[5]))

    print('\n')

    print('---------------')
    print('SAMPLER: ROS')
    print('SETTING: S1')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[6]))

    print('---------------')
    print('SAMPLER: ROS')
    print('SETTING: S2')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[7]))

    print('---------------')
    print('SAMPLER: ROS')
    print('SETTING: S3')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[8]))

    print('\n')

    print('---------------')
    print('SAMPLER: SMOTE')
    print('SETTING: S1')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[9]))

    print('---------------')
    print('SAMPLER: SMOTE')
    print('SETTING: S2')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[10]))

    print('---------------')
    print('SAMPLER: SMOTE')
    print('SETTING: S3')
    print('METRIC:', name)
    print('---------------')
    print('{:.2f}'.format(metric[11]))

    print('\n')

