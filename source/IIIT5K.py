import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class IIIT5K(Dataset):
    def __init__(self, data_dir="./data/IIIT5K/", is_train=True, transform=None, image_size=(3, 32, 400)):
        self.data_dir = data_dir
        train_mat_file_path = "trainCharBound.mat"
        test_mat_file_path = "testCharBound.mat"
        self.traindata = scio.loadmat(os.path.join(data_dir, train_mat_file_path))['trainCharBound'][0]
        self.testdata = scio.loadmat(os.path.join(data_dir, test_mat_file_path))['testCharBound'][0]
        if is_train:
            self.data = self.traindata
        else:
            self.data = self.testdata
        self.transform = transform
        self.image_size = image_size
        self.max_length, self.num_classes, self.char_to_id, self.id_to_char = self._init_data(self.traindata)
    
    def __len__(self):
        return len(self.data)

    def _init_data(self, data):
        data_set = "卍"
        max_length = 0
        for each_data in data: 
            label = each_data[1][0]
            length = len(label)
            if length > max_length:
                max_length = length
            
            for c in label:
                if c not in data_set:
                    data_set += c
        
        num_classes = len(data_set)
        char_to_id = {j:i for i, j in enumerate(data_set)}
        id_to_char = {i:j for i, j in enumerate(data_set)}
        return max_length, num_classes, char_to_id, id_to_char

    def __getitem__(self, idx):
        image_name = self.data[idx][0][0]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)
        image = image.resize((self.image_size[2], self.image_size[1]), Image.ANTIALIAS)
        image = np.array(image)
        if self.image_size[0] == 3:
            if len(image.shape) == 2:
                image = np.tile(image, (3, 1, 1))
            else:
                image = image.transpose((2, 0, 1))
            d_label = self.data[idx][1][0]
        label = np.zeros((self.max_length))
        label[:len(d_label)] = [int(self.char_to_id[c]) for c in d_label]
        #label = np.array([int(self.char_to_id[c]) for c in d_label])
        input_length = np.array((24))
        target_length = np.array((len(d_label)))
        sample = {'image': image,
                'label': label,
                'input_length': input_length,
                'target_length': target_length}
        return sample

if __name__ == "__main__":
    dataset = IIIT5K()
    print(len(dataset))
    dataLoader = DataLoader(dataset, 10, shuffle=True, num_workers=1)
    for i, bacth_data in enumerate(dataLoader):
        print(i, bacth_data['image'].shape)