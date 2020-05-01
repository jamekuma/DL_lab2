import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Cifar10Dataset(Dataset):
    def __init__(self, root_path, train=True):
        super(Cifar10Dataset, self).__init__()
        self.datas = None
        self.labels = []
        train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_batches = ['test_batch']
        if train:
            batches = train_batches
        else :
            batches = test_batches
        for file_name in batches:
            with open(root_path + file_name, 'rb') as file:
                dict = pickle.load(file, encoding='latin1')
                datas = dict['data'].reshape(-1, 3, 32, 32)
                labels = dict['labels']
                if self.datas is None:
                    self.datas = datas
                else:
                    self.datas = np.append(self.datas, datas, axis=0)
                self.labels += labels
        self.datas = torch.from_numpy(self.datas).type(torch.FloatTensor)


    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)


