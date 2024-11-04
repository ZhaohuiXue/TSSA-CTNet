import numpy as np
import os
from torch.utils.data import Dataset
import json

def read_json_file(json_file_path):
     with open(json_file_path, 'r') as file:
         data = json.load(file)
     return data

def labelcovert(label):
    if label == 0:
        return 0, 1, 0
    elif label == 1:
        return 1, 2, 0
    elif label == 2:
        return 2, 3, 0
    elif label == 3:
        return 3, 0, 1
    elif label == 4:
        return 4, 0, 2
    elif label == 5:
        return 5, 0, 3
    elif label == 6:
        return 6, 0, 0

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    return data

def interdata(input):
    if True in np.isnan(input):
        t, f, h, w = input.shape
        input_t = input.transpose(1,2,3,0).reshape(-1, t)
        input_inter = []
        for i in range(f * h * w):
            input_t_i = input_t[i]
            if np.isnan(input_t_i).any():
                missing_indexes = np.isnan(input_t_i)
                new_indexes = np.arange(len(input_t_i))
                input_t_i = np.interp(new_indexes, new_indexes[~missing_indexes], input_t_i[~missing_indexes])
            input_inter.append(input_t_i)
        input_inter = np.array(input_inter).reshape(f, h, w, t).transpose(3, 0, 1, 2)
        return input_inter
    else:
        return input

def normalize8(data):
    max_val = 65535
    min_val = 4894
    data = (data - min_val) / (max_val - min_val)
    return data

def normalize7(data):
    max_val = 255
    min_val = 1
    data = (data - min_val) / (max_val - min_val)
    return data



class SemiCDDataset(Dataset):
    def __init__(self, root, mode, data_lis, temporal, patch_size):
       
        self.root = root
        self.mode = mode
        self.patch = int(patch_size//2)
        
        self.tm1 = str(temporal[0])
        self.tm2 = str(temporal[1])
            
        self.ids = data_lis
            

    def __getitem__(self, item):
        id = self.ids[item]
        
        imgA = np.load(os.path.join(self.root, self.tm1, id['path']))#[:, :, 5-self.patch: 5+self.patch+1, 5-self.patch: 5+self.patch+1]
          
        imgB = np.load(os.path.join(self.root, self.tm2, id['path']))#[:, :, 5-self.patch: 5+self.patch+1, 5-self.patch: 5+self.patch+1]

        #imgA = interdata(imgA)
        #imgA = normalize7(imgA)

        #imgB = interdata(imgB)
        #imgB = normalize8(imgB)

        #imgA = np.nan_to_num(imgA, nan=0)
        #imgB = np.nan_to_num(imgB, nan=0)
        
        label = int(id['label'])
        label_c, labelA, labelB = labelcovert(label)
 
        #if self.mode == 'train':
            #imgA, imgB, maskA, maskB = resize(imgA, imgB, maskA, maskB, (0.8, 1.2))
            #imgA, imgB, maskA, maskB = crop(imgA, imgB, maskA, maskB, self.size)
            #imgA, imgB, maskA, maskB = hflip(imgA, imgB, maskA, maskB, p=0.5)
            #return imgA, imgB, maskA, maskB

        return imgA, imgB, label_c, labelA, labelB

    def __len__(self):
        return len(self.ids)
