from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
h5py._errors.unsilence_errors()
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime
from tqdm.notebook import tqdm



class MyDataset(Dataset):

    def __init__(
            self,
            data_filenames,
            augmentation=None, 
            preprocessing=None
    ):

        # self.scaling_dict = scaling_dict
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        data_dict = self._load_datafiles(data_filenames)
        
        scaling_dict=self._calculate_scaling_dict(data_dict)
        
        self.image, self.mask = self._preprocess_data_cube(data_dict, scaling_dict)
        #print(self.image.shape, self.mask.shape)
        self.data_len = self.image.shape[-1]
        
        
    def __len__(self):
        # assume one file for now
        return self.data_len # last element we cann't predict

    def __getitem__(self, idx):
        
        image = self.image[:, :, :, idx]
        mask = self.mask[:, :, idx]
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask


    def _calculate_scaling_dict(self, data_list, C_scaling=100):
                # estimate sample mean and std -- this should be done better
        n_training_samples=len(data_list['C'])
        
        C = np.stack([data_list['C'][idx] for idx in range(n_training_samples)])
        eps = np.stack([data_list['eps'][idx] for idx in range(n_training_samples)])
        Ux = np.stack([data_list['Ux'][idx] for idx in range(n_training_samples)])
        Uy = np.stack([data_list['Uy'][idx] for idx in range(n_training_samples)])

        Ux_mean, Ux_std = Ux.mean(), Ux.std()
        Uy_mean, Uy_std = Uy.mean(), Uy.std()
        eps_mean, eps_std = eps.mean(), eps.std()


        # print(Ux_mean, Ux_std)
        # print(Uy_mean, Uy_std)
        # print(eps_mean, eps_std)

        data_scalingdict = {
            'C_scaling': C_scaling,
            'Ux_mean': Ux_mean,
            'Ux_std': Ux_std,
            'Uy_mean': Uy_mean,
            'Uy_std': Uy_std,
            'eps_mean': eps_mean,
            'eps_std': eps_std,
        }
        

        return data_scalingdict
        
        
        
    ## add helper function
    def _read_simulation_hdf(self, file_name):
        #(f'loading the file: {file_name}')
        data_dict = {}

        with h5py.File(file_name, "r") as file_handle:
            # List all groups
            #print(f"Keys: {file_handle.keys()}")
            scaling_factor = 1
            for key_ in file_handle.keys():
                if 'key_' == 'C':
                    scaling_factor = 1 # 100
                elif 'key_' == 'Ux' or 'key_' == 'Uy':
                    scaling_factor = 1 # 1000
                
                data_dict[key_] = scaling_factor * np.array(file_handle[key_])
                #print(f'Done loading the variable {key_} of shape: {data_dict[key_].shape}')

            #print(f'Done with {file_name} == closing file now')

        return data_dict['C'], data_dict['eps'], data_dict['Ux'], data_dict['Uy'],




    def _preprocess_data_cube(self, data_dict, scaling_dict):
        #print(f'preprocess_data_cube')

        masks = []
        images = []
        for file_idx in range(len(data_dict['C'])):
            C = data_dict['C'][file_idx][:, :, :-1]
            eps = data_dict['eps'][file_idx][:, :, :-1]
            Ux = data_dict['Ux'][file_idx][:, :, :-1]
            Uy = data_dict['Uy'][file_idx][:, :, :-1]
            
            C_t = data_dict['eps'][file_idx][:, :, 1:]
            eps_t = data_dict['eps'][file_idx][:, :, 1:]
            Ux_t = data_dict['Ux'][file_idx][:, :, 1:]
            Uy_t = data_dict['Uy'][file_idx][:, :, 1:]
            
            # mask = log_transform(eps_t - eps[:, :, :-1]) # this scaled from 0 to 1
            
            #model baseline
            #mask = eps_t - eps
            
            #model II: predict next snapshot directly!
            mask = eps_t
            
            #Model III
            #mask = np.stack([C_t, eps_t, Ux_t, Uy_t], axis=-1)
            #mask = np.swapaxes(mask, 3, 2)
            
            # these should be moved to preprocessing
            # C_scaled = log_transform(C*scaling_dict['C_scaling']) - 0.5 # scale to be from 0 to 1
            C = C*scaling_dict['C_scaling'] - 0.5
            Ux = (Ux - scaling_dict['Ux_mean']) / scaling_dict['Ux_std']
            Uy = (Uy - scaling_dict['Uy_mean']) / scaling_dict['Uy_std']
            eps = (eps - scaling_dict['eps_mean']) / scaling_dict['eps_std']
                
            #wait, why are doing this??? concatenate?
            image = np.stack([C, eps, Ux, Uy], axis=-1)
            
            
            image = np.swapaxes(image, 3, 2)

            masks.append(mask)
            images.append(image)
        
        masks = np.concatenate(masks, axis=-1)
        images = np.concatenate(images, axis=-1)
        #print(f'preprocess_data_cube: {masks.shape}, {images.shape}')
        
        return images, masks

    def _load_datafiles(self, data_filenames):
        data_dict = {'C': [], 'eps': [], 'Ux': [], 'Uy': []}
        for filename in data_filenames:
            C, eps, Ux, Uy = self._read_simulation_hdf(filename)
            data_dict['C'].append(C[2:-2, 2:-2, :])
            data_dict['eps'].append(eps[2:-2, 2:-2, :])
            data_dict['Ux'].append(Ux[2:-2, 2:-2, :])
            data_dict['Uy'].append(Uy[2:-2, 2:-2, :])
        return data_dict
        
    
    



    



    
 
