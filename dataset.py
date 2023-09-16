import torch
import h5py
import numpy as np
import torch.nn as nn

class Quark_Gluon_Dataset(torch.utils.data.Dataset):
    def __init__(self,root_path,num_samples= 10000,transform=None):
        self.root_path = root_path
        self.transform = transform
        self.num_samples = num_samples
        self.f = h5py.File(self.root_path,'r')
        self.data_length = len(self.f['y'][:num_samples])

        self.X_jets = self.f['X_jets'][:num_samples]
        self.y = self.f['y'][:num_samples]

        if self.transform:
            self.X_jets = self.transform(torch.as_tensor(np.array(self.X_jets)).permute(0,3,2,1))
        else :
            self.X_jets = (self.X_jets - self.X_jets.min())/(self.X_jets.max()-self.X_jets.min())
    def __getitem__(self,idx):
        X_jets = self.X_jets[idx]
        # mass = f['m0'][:1000][idx]
        # momentum = f['pt'][:1000][idx]
        y = self.y[idx]

        return torch.as_tensor(np.array(X_jets)),torch.as_tensor(np.array(y))

    def __len__(self):
        return self.data_length