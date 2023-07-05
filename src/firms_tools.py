import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from bisect import bisect
from datetime import datetime
from netCDF4 import Dataset
from netCDF4 import num2date
from scipy import ndimage, interpolate
import gc
import cv2

from config import PRECISION, IMAGE_SIZE, SOUTH_AMERICA, FIRECCI_SHAPE


def scale_img(img, possible_max=1.0, possible_min=0.0):
    img = (img.astype(np.float64) - possible_min) / possible_max
    img = 255 * img
    img = img.astype(np.uint8)
    return img


def process_firms_chunk(f):
    cols_to_read = [
        "latitude",
        "longitude",
        "acq_date",
        "satellite",
        "instrument",
        "confidence",
    ]
    fire = pd.read_csv(
        f, usecols=cols_to_read, parse_dates=["acq_date"], low_memory=False
    )
    if fire.satellite.loc[0] in ["Terra", "Aqua", "N"]:
        fire.latitude = fire.latitude.round(PRECISION)
        fire.longitude = fire.longitude.round(PRECISION)
        fire.confidence = fire.confidence.replace({"l": 0, "n": 50, "h": 100})
        daily_fires = (
            fire.groupby(
                ["latitude", "longitude", "acq_date", "satellite", "instrument"]
            )
            .confidence.max()
            .reset_index()
        )
        daily_fires = daily_fires[
            daily_fires.confidence >= 50
        ]  # Remove low confidence records
        return daily_fires


def aggregate_month(fires_df):
    fires_df["year"] = fires_df.acq_date.dt.year
    fires_df["month"] = fires_df.acq_date.dt.month
    fires_df["ym"] = fires_df.acq_date.dt.strftime("%Y-%m")
    fires = (
        fires_df.groupby(["latitude", "longitude", "year", "month", "ym"])
        .size()
        .reset_index()
    )
    fires.columns = ["latitude", "longitude", "year", "month", "ym", "fire_cnt"]
    return fires


def get_lats_lons(region):
    # Resize to firecci shape
    M = np.load(f"maps/{region}-2020-08.npz")

    new_lats, new_lons = FIRECCI_SHAPE[region]

    x = np.arange(0, len(M["lats"]), 1) / (len(M["lats"]) - 1.)
    y = M["lats"]
    x_new = np.arange(0, new_lats, 1) / (new_lats - 1.)
    flats = interpolate.interp1d(x, y)
    lats = flats(x_new)

    x = np.arange(0, len(M["lons"]), 1) / (len(M["lons"]) - 1.)
    y = M["lons"]
    x_new = np.arange(0, new_lons, 1) / (new_lons - 1.)
    flons = interpolate.interp1d(x, y)
    lons = flons(x_new)

    return lats, lons


def get_tiles_df(region):
    lats, lons = get_lats_lons(region)
    tiles = []
    for i in range(0, lats.shape[0], IMAGE_SIZE):
        for j in range(0, lons.shape[0], IMAGE_SIZE):
            try:
                lat_min = lats[i]
                lat_max = lats[i + IMAGE_SIZE]
                lat_min, lat_max = sorted([lat_min, lat_max])
                lon_min = lons[j]
                lon_max = lons[j + IMAGE_SIZE]
                lon_min, lon_max = sorted([lon_min, lon_max])
                tiles.append(
                    [
                        i,
                        j,
                        lat_min,
                        lat_max,
                        lon_min,
                        lon_max,
                    ]
                )
            except Exception as e:
                pass

    tiles_df = pd.DataFrame(
        tiles,
        columns=[
            "i",
            "j",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
        ],
    )
    tiles_df["region"] = region
    return tiles_df


def rij_to_tile_id(region, i, j):
    return f'{region[:1]}-{i}-{j}'


def tile_id_to_rij(tile_id):
    r, i, j = tile_id.split('-')
    region = SOUTH_AMERICA
    return region, int(i), int(j)


def get_all_tile_info(use_cache=True):
    cache_path = "tile_info.csv"
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    tiles = get_tiles_df(SOUTH_AMERICA)
    tiles['tile_id'] = tiles.region.str[:1] + '-' + tiles.i.astype(str) + '-' + tiles.j.astype(str)
    tiles.to_csv(cache_path, index=False)
    return tiles


def get_tile_ids(lat, lon):
    tiles = get_all_tile_info()
    region, i, j = tiles.loc[
        (tiles.lat_min < lat) & \
        (tiles.lat_max >= lat) & \
        (tiles.lon_min < lon) & \
        (tiles.lon_max >= lon),
        ['region', 'i', 'j']
    ].values[0]
    return region, i, j


def find_coord_idx(value, coords):
    reverse = coords[0] > coords[-1]
    if reverse:
        coords = coords[::-1]
    if value < coords[0]:
        return -999
    if value > coords[-1]:
        return -999
    idx = bisect(coords, value)
    if reverse:
        idx = len(coords) - idx
    return idx - 1


def add_lat_lon_idx_to_fires(fires_df, tiles_df, lats, lons):
    fire_lats = fires_df[["latitude"]].drop_duplicates()
    fire_lats["one"] = 1
    fire_lats["lat_idx"] = fire_lats.latitude.apply(
        lambda v: find_coord_idx(v, lats.copy())
    )

    tile_lats = tiles_df[["i", "lat_min", "lat_max"]].drop_duplicates()
    tile_lats["one"] = 1

    lat_mapping = pd.merge(fire_lats, tile_lats, how="inner", on="one")
    lat_mapping = lat_mapping[
        (lat_mapping.latitude >= lat_mapping.lat_min)
        & (lat_mapping.latitude < lat_mapping.lat_max)
    ]
    lat_mapping = lat_mapping.drop(columns=["one", "lat_min", "lat_max"])

    fire_lons = fires_df[["longitude"]].drop_duplicates()
    fire_lons["one"] = 1
    fire_lons["lon_idx"] = fire_lons.longitude.apply(
        lambda v: find_coord_idx(v, lons.copy())
    )

    tile_lons = tiles_df[["j", "lon_min", "lon_max"]].drop_duplicates()
    tile_lons["one"] = 1

    lon_mapping = pd.merge(fire_lons, tile_lons, how="inner", on="one")
    lon_mapping = lon_mapping[
        (lon_mapping.longitude >= lon_mapping.lon_min)
        & (lon_mapping.longitude < lon_mapping.lon_max)
    ]
    lon_mapping = lon_mapping.drop(columns=["one", "lon_min", "lon_max"])

    fires_df = fires_df.merge(lat_mapping, how="inner", on="latitude")
    fires_df = fires_df.merge(lon_mapping, how="inner", on="longitude")
    return fires_df


def get_good_tiles(use_cache=True):
    cache_path = "good_tiles.csv"
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    sf_features = pd.concat(
        [pd.read_csv(f"features/{f}") for f in tqdm(os.listdir("features"))]
    )
    tiles = (
        sf_features[sf_features.layer.isin(["l1", "l2", "v1", "v2"])]
        .groupby(["region", "i", "j"])[["n_positive_pixels", "s0", "s1"]]
        .mean()
        .reset_index()
    )
    good_tiles = tiles[
        (tiles.s0 >= IMAGE_SIZE) & (tiles.s1 >= IMAGE_SIZE) & (
            tiles.n_positive_pixels > IMAGE_SIZE * IMAGE_SIZE / 5
            ) # at least 20% land surface
    ]
    good_tiles.to_csv(cache_path, index=False)
    return good_tiles


def create_tile_images(
        ds_name, region, y, m,
        sat=True, firms=True, burn=True
):
    ym_end = datetime(y, m, 1).strftime("%Y-%m")
    ym_start = datetime(y - 8, m, 1).strftime("%Y-%m")
    next_month = 1 if m == 12 else m + 1

    img_dir = f"data/{ds_name}/{region}/{ym_end}"
    os.makedirs(img_dir, exist_ok=True)

    good_tiles = get_good_tiles()
    good_tiles = good_tiles[good_tiles.region == region]

    # SAR
    if sat:
        M = np.load(f"maps/{region}-{ym_end}.npz")
        n_lats, n_lons = FIRECCI_SHAPE[region]
        layers_to_check = ["s1", "s2", "s3", "l1", "l2", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        for layer in layers_to_check:
            X = cv2.resize(M[layer], (n_lons, n_lats))
            img_dir = f"data/{ds_name}/{region}/{ym_end}/{layer}"
            os.makedirs(img_dir, exist_ok=True)
            for i, j in good_tiles[["i", "j"]].values:
                f = f"{img_dir}/{layer}-{ym_end}-{i}-{j}.npy"
                if not os.path.exists(f):
                    img = X[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                    np.save(f, img)

    # FIRMS
    if firms:
        fires_on_map = pd.read_csv(f"{region}_firms.csv")
        past_fires = fires_on_map[fires_on_map.ym <= ym_end].copy()
        past_fires = past_fires[past_fires.ym >= ym_start].copy()
        past_fires['yearmonth_int'] = past_fires.year * 12 + past_fires.month
        seasonal_fires = past_fires[past_fires.month == next_month].copy()

        current_fires = past_fires[past_fires.ym == ym_end].copy()

        fire_cnt_pixels = (
            past_fires.groupby(["lat_idx", "lon_idx"]).ym.nunique().reset_index()
        )
        fire_cnt_pixels_seasonal = (
            seasonal_fires.groupby(["lat_idx", "lon_idx"]).ym.nunique().reset_index()
        )
        fire_latest_pixels = (
            past_fires.groupby(["lat_idx", "lon_idx"]).yearmonth_int.max().reset_index()
        )

        fire_current_pixels = (
            current_fires.groupby(["lat_idx", "lon_idx"]).fire_cnt.max().reset_index()
        )

        fire_cnt_map = np.zeros(FIRECCI_SHAPE[region])
        for i, j, v in fire_cnt_pixels.values:
            fire_cnt_map[i, j] = v
        fire_cnt_map = fire_cnt_map.clip(0, 4)
        fire_cnt_map = scale_img(fire_cnt_map, 4)

        fire_seasonal_map = np.zeros(FIRECCI_SHAPE[region])
        for i, j, v in fire_cnt_pixels_seasonal.values:
            fire_seasonal_map[i, j] = v
        fire_seasonal_map = fire_seasonal_map.clip(0, 3)
        fire_seasonal_map = scale_img(fire_seasonal_map, 3)

        fire_recency_map = np.zeros(FIRECCI_SHAPE[region])
        for i, j, v in fire_latest_pixels.values:
            fire_recency_map[i, j] = (y * 12 + m) - v + 1  # months since last fire
        fire_recency_map = fire_recency_map.clip(0, 60)
        fire_recency_map = scale_img(fire_recency_map, 60)

        fire_current_map = np.zeros(FIRECCI_SHAPE[region])
        for i, j, v in fire_current_pixels.values:
            fire_current_map[i, j] = v
        fire_current_map = fire_current_map.clip(0, 4)
        fire_current_map = scale_img(fire_current_map, 4)

        for layer in ["firms_cnt", "firms_seasonal", "firms_recency", "firms_current"]:
            img_dir = f"data/{ds_name}/{region}/{ym_end}/{layer}"
            os.makedirs(img_dir, exist_ok=True)
            for i, j in good_tiles[["i", "j"]].values:
                f = f"{img_dir}/{layer}-{ym_end}-{i}-{j}.npy"
                if layer == "firms_cnt":
                    img = fire_cnt_map[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                elif layer == "firms_seasonal":
                    img = fire_seasonal_map[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                elif layer == "firms_recency":
                    img = fire_recency_map[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                elif layer == "firms_current":
                    img = fire_current_map[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                np.save(f, img)

    # FIRE CCI
    if burn:
        fire_cci_path = f"firecci/pngs/firecci_{ym_end}-01.png"
        burned_area = cv2.imread(fire_cci_path)
        burned_area = burned_area[:, :, 0]
        burned_area = burned_area.clip(0, 100)
        burned_area = scale_img(burned_area, 100)

        layer = "burn"
        img_dir = f"data/{ds_name}/{region}/{ym_end}/{layer}"
        os.makedirs(img_dir, exist_ok=True)
        for i, j in good_tiles[["i", "j"]].values:
            f = f"{img_dir}/{layer}-{ym_end}-{i}-{j}.npy"
            img = burned_area[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
            np.save(f, img)

