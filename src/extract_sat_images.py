import os
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

import cv2
import glob
from dateutil.relativedelta import relativedelta

from multiearth_challenge.datasets import base_datasets as bd
from multiearth_challenge import tiff_file_tools as tft


DATA_PATH = 'data/multiearth2023-dataset-final/'
DP_PATH = './dp'
os.makedirs(DP_PATH, exist_ok=True)


SOURCES = [
    # train files
    ('ls8', 'landsat8_train'),
    ('s1', 'sent1_train'),
    ('s2', 'sent2_b1-b4_train'),
    ('s2', 'sent2_b5-b8_train'),
    ('s2', 'sent2_b9-b12_train'),

    # ('ls5', 'landsat5_train'),

    # test files
    # ('s1', 'sent1_deforestation_segmentation'),
    # ('s2', 'sent2_deforestation_segmentation'),
    # ('ls8', 'landsat8_deforestation_segmentation'),

    # ('ls5', 'landsat5_fire_segmentation'),
]


def key_string(lat, lon, date_str):
    return f"{round(lat, 2):.2f}_{round(lon, 2):.2f}_{date_str}"


def extract_all_satellite_images(source, ds_name, roi):
    dataset = bd.NetCDFDataset(
        netcdf_file=f"{DATA_PATH}{ds_name}.nc",
        data_filters=[],
        merge_bands=False,
    )

    print(f"{ds_name}: Number of dataset samples: {len(dataset)}")

    SOURCE_DIR = f"{DP_PATH}/{source}"
    os.makedirs(SOURCE_DIR, exist_ok=True)

    for i, d in tqdm(enumerate(dataset)):
        
        band = d['bands'][0]
        lat, lon = d['lat_lon'].round(2)
        source_date = pd.to_datetime(d['date'])

        m0 = datetime(source_date.year, source_date.month, 1)
        m1 = m0 + relativedelta(months=1)
        m2 = m0 + relativedelta(months=2)

        m3 = m0 + relativedelta(months=-1)
        m4 = m0 + relativedelta(months=-2)

        keys_to_check = [
            key_string(lat, lon, m0.strftime("%Y-%m-%d")),
            key_string(lat, lon, m1.strftime("%Y-%m-%d")),
            key_string(lat, lon, m2.strftime("%Y-%m-%d")),
            key_string(lat, lon, m3.strftime("%Y-%m-%d")),
            key_string(lat, lon, m4.strftime("%Y-%m-%d")),
        ]

        key_in_roi = len([k for k in keys_to_check if k in roi])
        
        # Search all source images in +-2 months 
        if key_in_roi > 0:
            img = d['image'][0]
            
            source_date = source_date.strftime("%Y-%m-%d")
            source_year = source_date[:4]
            os.makedirs(f"{SOURCE_DIR}/{band}/{source_year}/", exist_ok=True)
            source_key = f"{source}_{band}_{key_string(lat, lon, source_date)}"
        
            source_path = f"{SOURCE_DIR}/{band}/{source_year}/{source_key}.npy"
            if not os.path.exists(source_path):
                np.save(source_path, img)


def get_forest_target():
    rows = []
    for mode in [
        "train",
        # "target",
        ]:
        forest_target_files = glob.glob(f"data/multiearth2023-dataset-final/forest_{mode}/Defores*.tiff", recursive=True)
        print(mode, len(forest_target_files))

        for tiff_path in tqdm(forest_target_files):
            d = tft.parse_filename_parts(tiff_path)
            d['target_path'] = tiff_path
            d['mode'] = mode
            img = cv2.imread(tiff_path)
            img = img[:, :, 0]
            img = img.clip(0, 1)
            d['img_mean'] = img.mean()
            rows.append(d)
    forest_target = pd.DataFrame(rows)
    return forest_target


def get_fire_target():
    rows = []
    for mode in [
        # "train_old",
        "target",
        ]:
        fire_target_files = glob.glob(f"data/fire_{mode}/Fire*.tiff", recursive=True)
        print(mode, len(fire_target_files))
        for tiff_path in tqdm(fire_target_files):
            d = tft.parse_filename_parts(tiff_path)
            d['target_path'] = tiff_path
            d['mode'] = mode
            img = cv2.imread(tiff_path)
            img = img[:, :, 0]
            d['img_mean'] = img.mean()
            rows.append(d)
    fire_target = pd.DataFrame(rows)
    return fire_target


BALANCE = 1
FIRE = 0

if __name__ == "__main__":
    if FIRE:
        target = get_fire_target()
    else:
        target = get_forest_target()
    print(target.shape)

    if BALANCE:
        b = pd.read_csv("balanced_lat_lons_v2.csv")
        target = target.merge(b[['lat', 'lon']], how='inner', on=['lat', 'lon'])
        print(target.shape)

    roi = {
        key_string(r.lat, r.lon, r.date.strftime("%Y-%m-%d"))
        for _, r in target.iterrows()
    }
    print(f"#ROI: {len(roi)}")

    for source, ds_name in tqdm(SOURCES):
        extract_all_satellite_images(source, ds_name, roi)



