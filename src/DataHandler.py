import os
import PIL.Image as Image
import numpy as np
import pickle
import torch
import csv
import shutil
import collections

from torch.utils.data import Sampler
import torchvision.transforms as transforms

class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']

        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']    

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]

        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]  

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels 
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label

class DatasetResampler(object):
    
    def __init__(self, root, split_dir, split_type, sup_data, sup_label, k_shot, n_way, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']

        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        

        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]  

        each_cnt = ori_labels.count(ori_labels[0])   # 600, each label image count
        resampler = int(each_cnt / k_shot) # how much each support image should be multiplied, EX) 600/5

        resampled_data = sup_data * resampler # EX) sup_set * 120
        resampled_label = sup_label * resampler

        data.extend(resampled_data)
        mapped_labels.extend(resampled_label)

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels  
        self.out_name = out_name
        self.length = len(self.data)
        # print('Dataset Resampler test')

    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            # print(img.shape)
            return img, label, self.data[index]
        else:
            # print(img.shape)
            return img, label


class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query, clone_factor=1):
    # Assume 5shot 5 way 15 query
        
        self.n_iter = n_iter    # assume 400
        self.n_way = n_way      # 5
        self.n_shot = n_shot    # 5
        self.n_query = n_query  # 15
        self.clone_factor = clone_factor

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter): # iterates 400 times
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
            for c in classes:  # pick 5 classes here
                l = self.m_ind[c.item()] # Get the index of each class
                pos = torch.randperm(l.size()[0]) # randomly pick some index
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]]) # use 15 of them as query set

            batch = torch.cat(batch_gallery + batch_query)
            yield batch


class AnalyzeSampler(Sampler):

    def __init__(self, label, n_iter, n_class, n_base, n_same, n_diff, rand_sampling):
    # Assume 5shot 5 way 15 query
        assert n_class > 0

        self.n_iter = n_iter    # assume 400
        self.n_class = n_class
        self.n_base = n_base
        self.n_same = n_same
        self.n_diff = n_diff

        self.rand_sampling = rand_sampling
        self.batch = []

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64, [0~63]

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.batch:
            for i in range(self.n_iter): 
                yield self.batch[i]    
        else:
            for i in range(self.n_iter): # iterates 400 times
                batch_same = []
                batch_diff = []
                classes = torch.randperm(len(self.m_ind))[:self.n_class] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
                for i,c in enumerate(classes):  # pick classes here
                    l = self.m_ind[c.item()] # Get the index of each class, .item() makes tensor to scalar
                    pos = torch.randperm(l.size()[0]) # randomly pick some index
                    if i == 0:
                        batch_same.append(l[pos[:self.n_base]])
                        batch_same.append(l[pos[self.n_base : self.n_base + self.n_same]]) 
                    else:
                        batch_diff.append(l[pos[:self.n_diff]]) 
                batch = torch.cat(batch_same + batch_diff)
                if not self.rand_sampling:
                    self.batch.append(batch)
                yield batch

class TripletSampler(Sampler):

    def __init__(self, label, n_iter, n_batch, rand_sampling, n_base, n_same, n_diff, n_class=2):

        self.n_iter = n_iter   
        self.n_batch = n_batch
        self.n_class = n_class
        self.n_base = n_base
        self.n_same = n_same
        self.n_diff = n_diff

        self.rand_sampling = rand_sampling
        self.batch = []

        label = np.array(label)
        self.m_ind = [] # store the index of the label
        unique = np.unique(label)
        unique = np.sort(unique) # unique eliminate duplication, so the total number of training classes are 64, [0~63]

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind has a shape of (64,600), which represents 600 images of 64 classes.

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.batch:
            for i in range(self.n_iter): 
                yield self.batch[i]    
        else:
            for _ in range(self.n_iter):
                batch_anchor = []
                batch_positive = []
                batch_negative = []
                for _ in range(self.n_batch): # Triplet Batch
                    classes = torch.randperm(len(self.m_ind))[:self.n_class] # torch.randperm: creates a range of parameters in random order, EX) if 5 then [1,0,4,3,2]
                    for i,c in enumerate(classes):  # pick classes here
                        l = self.m_ind[c.item()] # Get the index of each class, .item() makes tensor to scalar
                        pos = torch.randperm(l.size()[0]) # randomly pick some index
                        if i == 0:
                            batch_anchor.append(l[pos[:self.n_base]])
                            batch_positive.append(l[pos[self.n_base : self.n_base + self.n_same]]) 
                        else:
                            batch_negative.append(l[pos[:self.n_diff]]) 
                batch = torch.cat(batch_anchor + batch_positive + batch_negative)

                if not self.rand_sampling:
                    self.batch.append(batch)

                yield batch


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 

def without_augment(size=84, enlarge=False):
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

def with_augment(size=84, disable_random_resize=False):
    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(), # default = 0.5
            # transforms.RandomRotation(degrees=(30,-30)),
            transforms.ToTensor(),
            normalize,
        ])

