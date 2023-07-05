import os
import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date
from multiprocessing import Pool
from tqdm import tqdm
import gc
from firms_tools import scale_img
import random

from config import SOUTH_AMERICA


def _get_dates(ds):
    times = ds.variables["time"]
    dates = num2date(times[:], times.units)
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    return dates


def convert_ncs_to_npy(task_name):
    data_dir = f"appeears_data/{task_name}"
    vegetation_path = f"{data_dir}/VNP13A1.001_500m_aid0001.nc"
    leaf_path = f"{data_dir}/MCD15A2H.061_500m_aid0001.nc"
    surface_path = f"{data_dir}/VNP09H1.001_500m_aid0001.nc"

    TIME_IDX = 0  # Use -1 for latest possible image

    # Surface Reflection
    surface_ds = Dataset(surface_path, mode="r")
    lats = surface_ds.variables["lat"][:].compressed()
    lons = surface_ds.variables["lon"][:].compressed()
    s1 = surface_ds.variables["SurfReflect_I1"]
    s2 = surface_ds.variables["SurfReflect_I2"]
    s3 = surface_ds.variables["SurfReflect_I3"]
    s_date = _get_dates(surface_ds)[TIME_IDX]
    smax = 1.2
    smin = -0.01
    s1_map = s1[TIME_IDX, :, :].filled(fill_value=smin).clip(smin, smax)
    s2_map = s2[TIME_IDX, :, :].filled(fill_value=smin).clip(smin, smax)
    s3_map = s3[TIME_IDX, :, :].filled(fill_value=smin).clip(smin, smax)
    surface_ds.close()
    del surface_ds, s1, s2, s3
    gc.collect()

    #Leaf coverage
    leaf_ds = Dataset(leaf_path, mode="r")
    l1 = leaf_ds.variables["Fpar_500m"]
    l2 = leaf_ds.variables["Lai_500m"]
    l_date = _get_dates(leaf_ds)[TIME_IDX]
    l1_map = l1[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    l2_map = l2[TIME_IDX, :, :].filled(fill_value=0).clip(0, 7)
    leaf_ds.close()
    del leaf_ds, l1, l2
    gc.collect()

    # Vegetation indices
    vegetation_ds = Dataset(vegetation_path, mode="r")
    v1 = vegetation_ds.variables["_500_m_16_days_EVI"]
    v2 = vegetation_ds.variables["_500_m_16_days_NDVI"]
    v3 = vegetation_ds.variables["_500_m_16_days_NIR_reflectance"]
    v4 = vegetation_ds.variables["_500_m_16_days_SWIR1_reflectance"]
    v5 = vegetation_ds.variables["_500_m_16_days_SWIR2_reflectance"]
    v6 = vegetation_ds.variables["_500_m_16_days_green_reflectance"]
    v7 = vegetation_ds.variables["_500_m_16_days_red_reflectance"]
    v_date = _get_dates(vegetation_ds)[TIME_IDX]
    v1_map = v1[TIME_IDX, :, :].filled(fill_value=-1).clip(-1, 1)
    v2_map = v2[TIME_IDX, :, :].filled(fill_value=-1).clip(-1, 1)
    v4_map = v4[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    v5_map = v5[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    v3_map = v3[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    v6_map = v6[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    v7_map = v7[TIME_IDX, :, :].filled(fill_value=0).clip(0, 1)
    vegetation_ds.close()
    del vegetation_ds, v1, v2, v3, v4, v5, v6, v7
    gc.collect()

    s1_map = scale_img(s1_map, possible_max=smax, possible_min=smin)
    s2_map = scale_img(s2_map, possible_max=smax, possible_min=smin)
    s3_map = scale_img(s3_map, possible_max=smax, possible_min=smin)
    l1_map = scale_img(l1_map, possible_max=1, possible_min=0)
    l2_map = scale_img(l2_map, possible_max=7, possible_min=0)
    v1_map = scale_img(v1_map, possible_max=1, possible_min=-1)
    v2_map = scale_img(v2_map, possible_max=1, possible_min=-1)
    v3_map = scale_img(v3_map, possible_max=1, possible_min=0)
    v4_map = scale_img(v4_map, possible_max=1, possible_min=0)
    v5_map = scale_img(v5_map, possible_max=1, possible_min=0)
    v6_map = scale_img(v6_map, possible_max=1, possible_min=0)
    v7_map = scale_img(v7_map, possible_max=1, possible_min=0)
    
    np.savez_compressed(
        f"maps/{task_name}.npz",
        s1=s1_map,
        s2=s2_map,
        s3=s3_map,
        l1=l1_map,
        l2=l2_map,
        v1=v1_map,
        v2=v2_map,
        v3=v3_map,
        v4=v4_map,
        v5=v5_map,
        v6=v6_map,
        v7=v7_map,
        lats=lats,
        lons=lons,
        s_date=s_date,
        l_date=l_date,
        v_date=v_date,
    )

    os.remove(vegetation_path)
    os.remove(leaf_path)
    os.remove(surface_path)


def update(res):
    pbar.update()


def error(e):
    print(e)


if __name__ == "__main__":
    os.makedirs("maps", exist_ok=True)

    # pool = Pool(processes=2)
    task_names = [
        task_name
        for task_name in os.listdir("appeears_data")
        if not os.path.exists(f"maps/{task_name}.npz")
    ]
    task_names = [f for f in task_names if f.startswith(SOUTH_AMERICA)]
    print(len(task_names))

    random.shuffle(task_names)
    print(task_names[:5])

    for task_name in tqdm(task_names):
        convert_ncs_to_npy(task_name)

    pbar = tqdm(total=len(task_names))

    # for task_name in task_names:
    #     pool.apply_async(
    #         convert_ncs_to_npy, args=(task_name,), callback=update, error_callback=error
    #     )

    # pool.close()
    # pool.join()
    # pbar.close()
