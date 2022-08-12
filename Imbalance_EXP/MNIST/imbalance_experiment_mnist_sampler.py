import imbalance_experiment_mnist_generator
from imbalance_experiment_mnist_generator import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from torch.utils.data import TensorDataset, DataLoader

def sampler_MNIST(setting, oversample, undersample, smote, seed):
    if setting == 'Half-Split':
        sample_size = 25050
        ros_sample_size = 50000
        rus_sample_size = 100
        trainloader = trainloader_mnist_1

    if setting == 'Multimajority':
        sample_size = 48605
        ros_sample_size = 54000
        rus_sample_size = 50
        trainloader = trainloader_mnist_2

    if setting == 'Multiminority':
        sample_size = 5845
        ros_sample_size = 58000
        rus_sample_size = 50
        trainloader = trainloader_mnist_3

    output_container_data = torch.tensor((), device=device)
    output_container_labels = torch.tensor((), device=device)

    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        output_container_data = torch.cat((output_container_data, data), 0)
        output_container_labels = torch.cat((output_container_labels, labels), 0)


    output_container_data_numpy = output_container_data.cpu().detach().numpy()
    output_container_data_numpy_reshaped = output_container_data_numpy.reshape(sample_size, 28*28)
    output_container_labels_numpy = output_container_labels.cpu().detach().numpy()

    if oversample==True:
        ros = RandomOverSampler(random_state=seed)
        sampled_data, sampled_labels = ros.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(ros_sample_size, 1, 28, 28)
    
    if undersample==True:
        rus = RandomUnderSampler(random_state=seed)
        sampled_data, sampled_labels = rus.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(rus_sample_size, 1, 28, 28)

    if smote==True:
        smote = SMOTE(random_state=seed, k_neighbors=3)
        sampled_data, sampled_labels = smote.fit_resample(output_container_data_numpy_reshaped, output_container_labels_numpy)
        sampled_data = sampled_data.reshape(ros_sample_size, 1, 28, 28)

    sampled_data = torch.from_numpy(sampled_data)
    sampled_labels = torch.from_numpy(sampled_labels)
    sampled_labels = sampled_labels.long()

    Resampled_Dataset = TensorDataset(sampled_data, sampled_labels)
    trainloader_new = DataLoader(Resampled_Dataset, batch_size=16, shuffle=True) 

    return trainloader_new


##Undersample Datasets for each Setting:
MNIST_S1_Undersample_Trainloader = sampler_MNIST('Half-Split', oversample=False, undersample=True, smote=False, seed=1)
MNIST_S2_Undersample_Trainloader = sampler_MNIST('Multimajority', oversample=False, undersample=True, smote=False, seed=2)
MNIST_S3_Undersample_Trainloader = sampler_MNIST('Multiminority', oversample=False, undersample=True, smote=False, seed=3)

##Oversample Datasets for each Setting:
MNIST_S1_Oversample_Trainloader = sampler_MNIST('Half-Split', oversample=True, undersample=False, smote=False, seed=1)
MNIST_S2_Oversample_Trainloader = sampler_MNIST('Multimajority', oversample=True, undersample=False, smote=False, seed=2)
MNIST_S3_Oversample_Trainloader = sampler_MNIST('Multiminority', oversample=True, undersample=False, smote=False, seed=3)

##SMOTE Datasets for each Setting:
MNIST_S1_SMOTE_Trainloader = sampler_MNIST('Half-Split', oversample=False, undersample=False, smote=True, seed=1)
MNIST_S2_SMOTE_Trainloader = sampler_MNIST('Multimajority', oversample=False, undersample=False, smote=True, seed=2)
MNIST_S3_SMOTE_Trainloader = sampler_MNIST('Multiminority', oversample=False, undersample=False, smote=True, seed=3)
