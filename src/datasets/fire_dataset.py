import collections.abc
import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        if cfg.filter_i_range:
            df = df[
                (df.i > cfg.filter_i_range[0]) & (df.i < cfg.filter_i_range[1])
            ]
            print(f"Filtered: {df.shape}")

        self.df = df.copy()
        # self.df['input_date'] = pd.to_datetime(self.df['input_date'])
        self.i_coords = self.df['i'].values 
        self.j_coords = self.df['j'].values
        # self.input_dates = self.df['input_date'].values
        self.input_ys = self.df['input_y'].values
        self.input_ms = self.df['input_m'].values
        
        self.aug = aug
    
    def get_tile_img_path(self, ym, layer, i, j):
        img_dir = f"{self.cfg.data_dir}/data/fs_256/sa/{ym}/{layer}"
        return f"{img_dir}/{layer}-{ym}-{i}-{j}.npy"

    def get_sat_img(self, ym, layer, i, j):
        source_path = self.get_tile_img_path(ym, layer, i, j)
        img = np.load(source_path)
        return img

    def get_input_tensor(self, input_date, i, j):
        chs = []
        for layer, (a, b) in self.cfg.layer_windows.items():
            for lag in range(a, b):
                lag_date = input_date + relativedelta(months = - lag)
                lag_ym = lag_date.strftime("%Y-%m")
                ch = self.get_sat_img(lag_ym, layer, i, j)
                chs.append(ch)
        x = np.stack(chs)
        x = x.transpose(1, 2, 0)
        return x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        i =self.i_coords[idx]
        j =self.j_coords[idx]
        # input_date = self.input_dates[idx]
        y = self.input_ys[idx]
        m = self.input_ms[idx]
        input_date = datetime(y, m, 1)


        img = self.get_input_tensor(input_date, i, j)
        img = img / 255.

        mask = self.get_mask(input_date, i, j)

        if self.aug:
            img, mask = self.augment(img, mask)

        
        feature_dict = {
            "input": torch.tensor(img.transpose(2, 0, 1)).float(),
            "mask": torch.tensor(mask),
        }
        return feature_dict

    def get_mask(self, input_date, i, j):
        target_date = input_date + relativedelta(months = self.cfg.target_lag)
        target_ym = target_date.strftime("%Y-%m")
        mask = self.get_sat_img(target_ym, "burn", i, j)
        mask = mask / 255.
        mask = 1 * (mask >= 0.5) 
        mask = mask.clip(0, 1)
        return mask


    def augment(self, img, mask):
        img = img.astype(np.float32)
        if mask is not None:
            transformed = self.aug(image=img, mask=mask)
            trans_img = transformed["image"]
            trans_mask = transformed["mask"]

            return trans_img, trans_mask
        else:
            transformed = self.aug(image=img)
            trans_img = transformed["image"]

            return trans_img, None


