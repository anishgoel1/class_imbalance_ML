
!pip install quo
!pip install pyperclip

import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from quo import echo

dataset = 'MNIST'
batch_size = 60

if dataset == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
    print('Datasets are being downloaded...')

    trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    class_list = ('0', '1', '2', '3', '4', '5','6', '7', '8', '9')
    
    class_number = len(class_list)
    print('Download finished!')

elif dataset == 'CIFAR-10':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print('Datasets are being downloaded...')

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    class_list = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    class_number = len(class_list)
    print('Download finished!')

elif dataset == 'CIFAR-100':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print('Datasets are being downloaded...')

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    class_list = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 
                  'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 
                  'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 
                  'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                  'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 
                  'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
                  'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    
    class_number = len(class_list)
    print('Download finished!')

if dataset == 'MNIST':
    for item, class_n in zip(torch.bincount(trainset.targets), class_list):
        print("Class", class_n ,":", np.array(item))

    min_number = np.array(min(torch.bincount(trainset.targets)))
    print("Smallest number of examples in one of the classes:", min_number)

train_targets = np.array(trainset.targets)
_, train_class_counts = np.unique(train_targets, return_counts=True)

test_targets = np.array(testset.targets)
_, test_class_counts = np.unique(test_targets, return_counts=True)

train_output = np.all(train_class_counts == train_class_counts[0]) 

if train_output:
    echo(f'Training set is balanced', bold=True)
else:
    echo(f'Training set is imbalanced', bold=True)

    rho_train = np.max(train_class_counts) / np.min(train_class_counts)
    print('The rho value is', rho_train) 

    mu_train = ((train_class_counts == np.min(train_class_counts)).sum()) / train_class_counts.size
    print('The mu value is', mu_train) 

    if ((train_class_counts == np.min(train_class_counts)).sum()) > ((train_class_counts == np.max(train_class_counts)).sum()):
        print('More minority classes than majority classes')

    if ((train_class_counts == np.max(train_class_counts)).sum()) > ((train_class_counts == np.min(train_class_counts)).sum()):
        print('More majority classes than minority classes')

    if ((train_class_counts == np.max(train_class_counts)).sum()) == ((train_class_counts == np.min(train_class_counts)).sum()):
        print('Same number of minority and majority classes')

    if ((train_class_counts == np.min(train_class_counts)).sum()) > ((train_class_counts.size) - ((train_class_counts == np.min(train_class_counts)).sum())):
        print('Predominantly minority class')

    if ((train_class_counts == np.max(train_class_counts)).sum()) > ((train_class_counts.size) - ((train_class_counts == np.max(train_class_counts)).sum())):
        print('Predominantly majority class')

test_output = np.all(test_class_counts == test_class_counts[0]) 

if train_output:
    echo(f'\nTest set is balanced', bold=True)
else:
    echo(f'\nTest set is imbalanced', bold=True)

    rho_test = np.max(test_class_counts) / np.min(test_class_counts)
    print('The rho value is', rho_test) 

    mu_test = ((test_class_counts == np.min(test_class_counts)).sum()) / test_class_counts.size
    print('The mu value is', mu_test) 

    if ((test_class_counts == np.min(test_class_counts)).sum()) > ((test_class_counts == np.max(test_class_counts)).sum()):
        print('More minority classes than majority classes')

    if ((test_class_counts == np.max(test_class_counts)).sum()) > ((test_class_counts == np.min(test_class_counts)).sum()):
        print('More majority classes than minority classes')

    if ((test_class_counts == np.max(test_class_counts)).sum()) == ((test_class_counts == np.min(test_class_counts)).sum()):
        print('Same number of minority and majority classes')

    if ((test_class_counts == np.min(test_class_counts)).sum()) > ((test_class_counts.size) - ((test_class_counts == np.min(test_class_counts)).sum())):
        print('Predominantly minority class')

    if ((test_class_counts == np.max(test_class_counts)).sum()) > ((test_class_counts.size) - ((test_class_counts == np.max(test_class_counts)).sum())):
        print('Predominantly majority class')

plot1 = sns.relplot(x=class_list, y=train_class_counts, hue=class_list)

plot1.set(xticklabels=[])  
plot1.set(title='Training Class Distribution')  
plot1.set(xlabel=None)  
plt.xlabel('Class Name')
plt.ylabel('Class Count')
sns.move_legend(plot1, loc="upper right", bbox_to_anchor=(2, 0.95), 
                ncol=5, frameon=False)
plt.show()

plot2 = sns.relplot(x=class_list, y=test_class_counts, hue=class_list)

plot2.set(xticklabels=[])  
plot2.set(title='Test Class Distribution')
plot2.set(xlabel=None)
plot2._legend.remove()
plt.xlabel('Class Name')
plt.ylabel('Class Count')
plt.show()
