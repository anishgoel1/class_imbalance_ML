import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split

val_size = 0.2
batch_size = 32
device = torch.device('cuda')

transform_CIFAR_10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class_names_CIFAR10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
nb_classes_CIFAR10 = len(class_names_CIFAR10)

def get_labels_and_class_counts(labels_list):
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts

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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from torch.utils.data import TensorDataset, DataLoader


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

