import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split

val_size = 0.2
batch_size = 32

transform_MNIST = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
class_names_MNIST = ('0', '1', '2', '3', '4', '5', '6', '7','8', '9')
nb_classes_MNIST = len(class_names_MNIST)

def get_labels_and_class_counts(labels_list):
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts

class ImbalanceGeneratorMNIST(Dataset):
    def __init__(self, num_samples, root, train, download, transform):
        self.dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform_MNIST)
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

        class_indices = [np.where(targets == i)[0] for i in range(nb_classes_MNIST)]

        self.imbal_class_counts = [
            int(prop)
            for count, prop in zip(class_counts, self.num_samples)
        ]

        idxs = []
        for c in range(nb_classes_MNIST):
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

    trainset = ImbalanceGeneratorMNIST(num_samples_train, root='.', train=True, download=True, transform=transform_MNIST)
    testset = ImbalanceGeneratorMNIST(num_samples_test, root='.', train=False, download=True, transform=transform_MNIST)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda)

    return trainloader, testloader, trainset

##Half Split Imbalance
setting1_mnist = dataset_creator(1, np.hstack(([1000]*5, [5000]*5)), np.hstack(([150] * 5, [750] * 5)))
trainloader_mnist_1 = setting1_mnist[0]
testloader_mnist_1 = setting1_mnist[1]
trainset_mnist_1 = setting1_mnist[2]

##Multimajority
setting2_mnist = dataset_creator(2, np.hstack(([5400]*9, [54])), np.hstack(([800] * 9, [8])))
trainloader_mnist_2 = setting2_mnist[0]
testloader_mnist_2 = setting2_mnist[1]
trainset_mnist_2 = setting2_mnist[2]

##Multiminority
setting3_mnist = dataset_creator(3, np.hstack(([58]*9, [5800])), np.hstack(([8] * 9, [800])))
trainloader_mnist_3 = setting3_mnist[0]
testloader_mnist_3 = setting3_mnist[1]
trainset_mnist_3 = setting3_mnist[2]