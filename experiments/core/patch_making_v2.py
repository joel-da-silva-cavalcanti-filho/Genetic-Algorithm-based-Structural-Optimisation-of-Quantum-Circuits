import torch
from torch import tensor, LongTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from core.ansatz_simulation_class import AnsatzSimulation
from tqdm import tqdm
import numpy as np 

def sample_data(dataset, classes, sample_size):
    class_one = 0
    class_two = 0
    binary_dataset = []

    for image in tqdm(dataset, desc='sampling data'):
        if image[1] == classes[0] and class_one < sample_size:
            binary_dataset.append(image)
            class_one +=1 
        elif image[1] == classes[1] and class_two < sample_size:
            binary_dataset.append(image)
            class_two += 1
            
        if class_one == sample_size and class_two == sample_size:
            break
    
    return binary_dataset

def sampling(dataset, classes, sample_size):
    samples = dict(zip(classes,[0] * len(classes)))
    sampled_dataset = []
    
    for image in tqdm(dataset, desc='sampling...'):
        if image[1] in classes:
            sampled_dataset.append(image)
            samples[image[1]] += 1
        
        if sum(samples.values()) == len(classes) * sample_size:
            break
    
    return sampled_dataset

class SampledDataset4Training(Dataset):
    def __init__(self, dataset, target_classes, transform=None):
        images, labels = zip(*dataset)
        labels = [0 if label == target_classes[0] else 1 for label in tqdm(labels, desc='labeling')]
        self.labels = tensor(labels)
        self.images = torch.stack(images)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        
        if self.transform:
            sample = self.transform(self.images[index])
        else:
            sample = self.images[index]
        
        
        return sample, self.labels[index]

class PatchExtraction(object):
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
    
    def __call__(self, image):
        unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        image_patches = unfold(image)
        #print(f'esse é a dimensão da imagem no mnist: {image.shape}')
        if image.shape[0] > 1:
            #print('voce nn possui três canais bro')
            n_channels, height, width = image.shape
            patch_len, n_patches = image_patches.shape
            image_patches = image_patches.view((n_channels, n_patches, int(patch_len/n_channels)))
            return image_patches
        else:
            patch_len, n_patches = image_patches.shape
            image_patches = image_patches.view((n_patches, patch_len))
            return image_patches
        
	
