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
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def key_string(lat, lon, date_str):
    return f"{round(lat, 2):.2f}_{round(lon, 2):.2f}_{date_str}"


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        if mode == "train" and cfg.balanced:
            b = pd.read_csv(f"{cfg.data_dir}/balanced_lat_lons_v2.csv")
            self.df = self.df.merge(b[['lat', 'lon']], how='inner', on=['lat', 'lon'])
        
        if mode == "train" and cfg.select_month:
            self.df['m'] = pd.to_datetime(self.df['date']).dt.month
            self.df = self.df[self.df.m == cfg.select_month]

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['source_end_date'] = self.df['date'] + timedelta(days=cfg.input_day_delta)
        self.df['source_end_date'] = self.df['source_end_date'].dt.strftime("%Y-%m-%d")

        self.df['source_start_date'] = self.df['date'] + timedelta(days = -60)
        self.df['source_start_date'] = self.df['source_start_date'].dt.strftime("%Y-%m-%d")
        self.df['date'] = self.df['date'].dt.strftime("%Y-%m-%d")


        print(self.df.shape)
        print(self.df.head())
        if "img_mean" in self.df.columns:
            print(self.df.img_mean.mean())
        self.target_paths = self.df.target_path.values

        self.sat_paths = self.get_sat_paths()  # extracted single channel images
        
        self.aug = aug

        self.candidates = self.get_candidates()

        self.sat_meta_dict = {
            target_path: df for target_path, df in self.candidates.groupby('target_path')
        }

        input_dates_check = self.candidates.groupby('target_path').source_date.count().mean()
        print(f"#input dates: {input_dates_check:.1f} df shape {df.shape}")


    def get_sat_paths(self):
        rows = []
        for source in self.cfg.sources:
            fs = glob.glob(f'{self.cfg.data_dir}/dp/{source}/**/*.npy', recursive=True)
            print((source, len(fs)))
            for f in fs:
                f_parts = f[:-4].split('/')[-1].split('_')
                source = f_parts[0]
                band = f_parts[-4]
                lat = round(float(f_parts[-3]), 2)
                lon = round(float(f_parts[-2]), 2)
                source_date = f_parts[-1]
                rows.append([source, band, lat, lon, source_date])
        df = pd.DataFrame(rows, columns=['source', 'band', 'lat', 'lon', 'source_date'])
        return df
    
    def get_candidates(self):
        source_coords = self.sat_paths.groupby([
            'source', 'lat', 'lon', 'source_date']).band.count().reset_index()
        candidates = self.df.merge(source_coords, on=['lat', 'lon'], suffixes = ['_target', '_source'])

        candidates = candidates[candidates.source_end_date > candidates.source_date] # sat image filter
        candidates = candidates[candidates.source_start_date < candidates.source_date] # sat image filter

        candidates = candidates.sort_values(by=['lat', 'lon', 'date', 'source', 'source_date'])
        candidates['source_rank'] =  candidates.groupby(['lat', 'lon', 'date', 'source']).source_date.rank(ascending=False)
        latest = candidates[candidates.source_rank <= self.cfg.max_rank]
        return latest

    def get_sat_img(self, source, band, lat, lon, source_date):
        source_year = source_date[:4]
        source_key = f"{source}_{band}_{key_string(lat, lon, source_date)}"
        source_path = f"{self.cfg.data_dir}/dp/{source}/{band}/{source_year}/{source_key}.npy"
        
        x = np.load(source_path)

        band_min, band_max = self.cfg.BAND_LIMITS[source][band]
        
        if self.cfg.normalization == "minmax":
            x = x.clip(band_min, band_max)
            if x.max() > x.min():
                x = (x - x.min()) / (x.max() - x.min())
            else:
                x = np.zeros((256, 256))
        elif self.cfg.normalization == "band":
            x = (x - band_min) / (band_max - band_min)
        elif self.cfg.normalization == "std":
            x = x.clip(band_min, band_max)
            if x.max() > x.min():
                x = (x - x.mean()) / x.std()
            else:
                x = np.zeros((256, 256))

        x = cv2.resize(x, dsize=(256, 256))

        return x

    def get_input_tensor(self, df):
        chs = []
        for source in self.cfg.sources:
            for source_rank in range(1, self.cfg.max_rank + 1):
                row = df[(df.source == source) & (df.source_rank == source_rank)]
                for band in self.cfg.BAND_LIMITS[source].keys():
                    if len(row) == 0:
                        ch = np.zeros((256, 256))
                    else:
                        ch = self.get_sat_img(source, band, row.lat.values[0], row.lon.values[0], row.source_date.values[0])
                    chs.append(ch)
        x = np.stack(chs)
        x = x.transpose(1, 2, 0)
        return x



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_path = self.target_paths[idx]
        
        if target_path in self.sat_meta_dict:
            df = self.sat_meta_dict[target_path].copy()
            if self.mode == "train" and self.cfg.source_date_dropout > 0.0 and len(df) > 1:
                if np.random.rand() < self.cfg.source_date_dropout:
                    n = np.random.randint(len(df) // 2, len(df))  # sample satellite dates
                    df = df.sample(n=n)
                    df['source_rank'] = df.groupby(['lat', 'lon', 'date', 'source']).source_date.rank(ascending=False)
        else:
            print(f'Missing input data: {target_path}')
            df = pd.DataFrame([], columns=['source', 'source_rank'])
        img = self.get_input_tensor(df)

        mask = self.get_mask(target_path)

        if self.aug:
            img, mask = self.augment(img, mask)

        # img = img / 255.
        feature_dict = {
            "input": torch.tensor(img.transpose(2, 0, 1)).float(),
            "mask": torch.tensor(mask),
        }
        return feature_dict

    def get_mask(self, target_path):
        mask = cv2.imread(f"{self.cfg.data_dir}{target_path}")[:, :, 0]
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


