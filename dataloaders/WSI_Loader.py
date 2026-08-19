import os

import h5py
import openslide
import pandas as pd
import torch
from torchvision.transforms import transforms
from torch.utils import data
from torch.utils.data import Dataset
from einops import rearrange
from PIL import Image

Image.MAX_IMAGE_PIXELS = 4083771720


class Camelyon16Dataset(Dataset):
    def __init__(self, csv_file, h5_dir, patch_size=256, region_size=4096, transform=None, is_train=False,
                 is_val=False):
        self.h5_dir = h5_dir
        self.transform = transform
        self.scale_level = 0
        self.region_size = region_size
        self.patch_size = patch_size
        self.is_train = is_train
        self.is_val = is_val
        self.wsi_image_paths = []
        self.csv_file = csv_file
        self.label_map = {
            'normal': 0,
            'tumor': 1
        }

        df = pd.read_csv(self.csv_file)
        for _, row in df.iterrows():
            if is_train:
                img_path = row['train']
            elif is_val:
                img_path = row['val']
            else:
                img_path = row['test']
            if img_path != '':
                self.wsi_image_paths.append(img_path)

    def __len__(self):
        return len(self.wsi_image_paths)

    def __getitem__(self, idx):
        image_name = os.path.basename(str(self.wsi_image_paths[idx]))
        image_name_final = os.path.basename(str(self.wsi_image_paths[idx]))
        h5_path = os.path.join(self.h5_dir, image_name.replace('.tif', '.h5'))
        wsi_image_path = self.wsi_image_paths[idx]
        wsi = openslide.open_slide(wsi_image_path)
        with h5py.File(h5_path, 'r') as h5_file:
            coords = h5_file['coords'][:]

        region_tensors = []
        for coord in coords:
            # Read 4096x4096 region
            region = wsi.read_region(coord, self.scale_level, (self.region_size, self.region_size)).convert('RGB')

            if self.transform:
                region = self.transform(region)

            # Split into 256x256 patches
            region = region.unsqueeze(0)
            region = region.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            region = rearrange(region, 'b c p1 p2 h w -> b (p1 p2) c h w')
            region_tensors.append(region)

        region_tensor = torch.cat(region_tensors, dim=0)
        label_str = image_name.split('_')[0]
        label = torch.tensor(self.label_map[label_str]).long()
        return region_tensor, label, image_name_final  # shape: [1, num_regions, 256, 3, 256, 256], label


def camelyon16_dataloader(h5_dir, csv_file, is_train=False, is_val=False):
    transform = transforms.Compose([
        transforms.Resize((4096, 4096)),
        transforms.ToTensor(),
    ])

    my_dataset = \
        Camelyon16Dataset(csv_file=csv_file, h5_dir=h5_dir, transform=transform, is_train=is_train, is_val=is_val)
    train_loader = data.DataLoader(my_dataset, batch_size=1, shuffle=True,
                                   num_workers=2, drop_last=False)
    return train_loader


class TCGALungDataset(Dataset):
    def __init__(self, h5_dir, wsi_dir, patch_size=256, region_size=4096, transform=None):
        self.h5_dir = h5_dir
        self.h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
        self.transform = transform
        self.scale_level = 0
        self.region_size = region_size
        self.patch_size = patch_size
        self.label_map = {
            'LUAD': 0,
            'LUSC': 1
        }
        self.wsi_dir = wsi_dir

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = os.path.join(self.h5_dir, self.h5_files[idx])
        h5_name = os.path.basename(str(self.h5_files[idx]))
        image_name = h5_name.replace('.h5', '.svs')
        image_name_final = str(h5_name.replace('.h5', '.svs'))

        wsi_image_path = os.path.join(self.wsi_dir, image_name)
        wsi = openslide.open_slide(wsi_image_path)
        with h5py.File(h5_path, 'r') as h5_file:
            coords = h5_file['coords'][:]

        region_tensors = []
        for coord in coords:
            # Read 4096x4096 region
            region = wsi.read_region(coord, self.scale_level, (self.region_size, self.region_size)).convert('RGB')

            if self.transform:
                region = self.transform(region)

            # Split into 256x256 patches
            region = region.unsqueeze(0)
            region = region.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            region = rearrange(region, 'b c p1 p2 h w -> b (p1 p2) c h w')
            region_tensors.append(region)

        region_tensor = torch.cat(region_tensors, dim=0)
        label_str = image_name.split('_')[0]
        label = torch.tensor(self.label_map[label_str]).long()
        return region_tensor, label, image_name_final  # shape: [1, num_regions, 256, 3, 256, 256], label


def tcga_lung_dataloader(h5_dir, wsi_dir):
    transform = transforms.Compose([
        transforms.Resize((4096, 4096)),
        transforms.ToTensor(),
    ])

    my_dataset = \
        TCGALungDataset(h5_dir=h5_dir, wsi_dir=wsi_dir, transform=transform)
    train_loader = data.DataLoader(my_dataset, batch_size=1, shuffle=True,
                                   num_workers=2, drop_last=False)
    return train_loader


class UBCOCEANDataset(Dataset):
    def __init__(self, h5_dir, wsi_dir, patch_size=256, region_size=4096, transform=None):
        self.h5_dir = h5_dir
        self.h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
        self.transform = transform
        self.scale_level = 0
        self.region_size = region_size
        self.patch_size = patch_size
        self.label_map = {
            'CC': 0,
            'EC': 1,
            'HGSC': 2,
            'LGSC': 3,
            'MC': 4,
        }
        self.wsi_dir = wsi_dir

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = os.path.join(self.h5_dir, self.h5_files[idx])
        h5_name = os.path.basename(str(self.h5_files[idx]))
        image_name = h5_name.replace('.h5', '.png')
        image_name_final = str(h5_name.replace('.h5', '.png'))

        wsi_image_path = os.path.join(self.wsi_dir, image_name)
        wsi = openslide.open_slide(wsi_image_path)
        with h5py.File(h5_path, 'r') as h5_file:
            coords = h5_file['coords'][:]

        region_tensors = []
        for coord in coords:
            # Read 4096x4096 region
            region = wsi.read_region(coord, self.scale_level, (self.region_size, self.region_size)).convert('RGB')

            if self.transform:
                region = self.transform(region)

            # Split into 256x256 patches
            region = region.unsqueeze(0)
            region = region.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            region = rearrange(region, 'b c p1 p2 h w -> b (p1 p2) c h w')
            region_tensors.append(region)

        region_tensor = torch.cat(region_tensors, dim=0)
        label_str = image_name.split('_')[0]
        label = torch.tensor(self.label_map[label_str]).long()
        return region_tensor, label, image_name_final


def ubc_ocean_dataloader(h5_dir, wsi_dir):
    transform = transforms.Compose([
        transforms.Resize((4096, 4096)),
        transforms.ToTensor(),
    ])

    my_dataset = \
        UBCOCEANDataset(h5_dir=h5_dir, wsi_dir=wsi_dir, transform=transform)
    train_loader = data.DataLoader(my_dataset, batch_size=1, shuffle=True,
                                   num_workers=2, drop_last=False)
    return train_loader


class WSITensorDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = torch.load(file_path)
        embedding = data['tensor']
        label = data['label']
        return embedding, label


def wsi_tensor_loader(tensors_dir):
    wsi_tensor_dataset = \
        WSITensorDataset(folder_path=tensors_dir)
    data_loader = data.DataLoader(wsi_tensor_dataset, batch_size=1, shuffle=True,
                                  num_workers=2, drop_last=False)
    return data_loader
