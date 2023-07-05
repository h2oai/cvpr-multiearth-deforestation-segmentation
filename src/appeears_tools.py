from matplotlib import dates
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date


def get_appeears_token():
    with open("appears.secret", "r") as infile:
        appears_login = json.load(infile)

    appears_auth = (appears_login["user"], appears_login["pswd"])
    response = requests.post(
        "https://appeears.earthdatacloud.nasa.gov/api/login", auth=appears_auth
    )
    token_response = response.json()
    return token_response["token"]


def list_appeears_tasks(token):
    response = requests.get(
        "https://appeears.earthdatacloud.nasa.gov/api/task",
        headers={"Authorization": "Bearer {0}".format(token)},
    )
    task_response = response.json()

    tasks = pd.DataFrame(task_response)
    tasks = tasks.sort_values(by="created", ascending=False)
    tasks = tasks[
        ["task_name", "task_id", "created", "completed", "estimate", "status"]
    ]
    return tasks


def get_appeears_files(token, task_id):
    response = requests.get(
        "https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}".format(task_id),
        headers={"Authorization": "Bearer {0}".format(token)},
    )
    bundle_response = response.json()
    files = pd.DataFrame(bundle_response["files"])
    return files.sort_values(by="file_size", ascending=False)


def download_appeears_file(token, task_id, file_id, filepath):
    response = requests.get(
        "https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}".format(
            task_id, file_id
        ),
        headers={"Authorization": "Bearer {0}".format(token)},
        allow_redirects=True,
        stream=True,
    )
    with open(filepath, "wb") as f:
        for data in response.iter_content(chunk_size=8192):
            _ = f.write(data)


def request_appeears_task(token, start_date, end_date, coordinates, task_name, mode="sat"):
    if mode == "sat":
        layers = [
            {"layer": "SurfReflect_I1", "product": "VNP09H1.001"},
            {"layer": "SurfReflect_I2", "product": "VNP09H1.001"},
            {"layer": "SurfReflect_I3", "product": "VNP09H1.001"},
            {"layer": "Lai_500m", "product": "MCD15A2H.061"},
            {"layer": "Fpar_500m", "product": "MCD15A2H.061"},
            {"layer": "_500_m_16_days_EVI", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_NDVI", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_NIR_reflectance", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_SWIR1_reflectance", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_SWIR2_reflectance", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_red_reflectance", "product": "VNP13A1.001"},
            {"layer": "_500_m_16_days_green_reflectance", "product": "VNP13A1.001"},
        ]
    elif mode == "lst":
        layers = [
            {"layer": "LST_Day_1KM", "product": "MYD21A2.061"},
            {"layer": "LST_Night_1KM", "product": "MYD21A2.061"},
        ] 
    req_json = _create_request_json(start_date, end_date, coordinates, layers, task_name)
    response = requests.post(
        "https://appeears.earthdatacloud.nasa.gov/api/task",
        json=req_json,
        headers={"Authorization": "Bearer {0}".format(token)},
    )
    task_response = response.json()
    return task_response


def _create_request_json(start_date, end_date, coordinates, layers, task_name):
    return {
        "params": {
            "geo": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": coordinates},
                        "properties": {},
                    }
                ],
                "fileName": "User-Drawn-Polygon",
            },
            "dates": [
                {
                    "endDate": end_date,
                    "recurring": False,
                    "startDate": start_date,
                    "yearRange": [1950, 2050],
                }
            ],
            "layers": layers,
            "output": {"format": {"type": "netcdf4"}, "projection": "geographic"},
        },
        "task_name": task_name,
        "task_type": "area",
    }



def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)


def get_monthly_latest_date(ds, prefix):
    times = ds.variables["time"]
    dates = num2date(times[:], times.units)
    dates = [[date.strftime('%Y-%m-%d'), date.strftime('%Y-%m')] for date in dates]
    dates_df = pd.DataFrame(dates, columns=['d', 'm'])
    dates_df.d = pd.to_datetime(dates_df.d)
    dates_df[f'date_idx'] = np.arange(len(dates_df))

    dates_df['last_day_of_month'] = dates_df.d.apply(last_day_of_month)
    dates_df['days_till_next_month'] = (dates_df.last_day_of_month - dates_df.d).dt.days
    dates_df = dates_df[dates_df.days_till_next_month > 15] 
    dates_df['latest'] = dates_df.groupby('m')['d'].rank(ascending=False)
    dates_df = dates_df[dates_df['latest'] == 1]
    dates_df = dates_df.drop(columns=['latest']).copy()
    # dates_df.d = dates_df.d.dt.strftime('%Y-%m-%d')

    dates_df = dates_df[['d', 'm', 'date_idx']]
    dates_df.columns = [f'{prefix}_d', 'm', f'{prefix}_date_idx']
    return dates_df


def get_dates_df(task_id):
    data_dir = f'appeears_data/{task_id}'
    vegetation_path = f'{data_dir}/VNP13A1.001_500m_aid0001.nc'
    leaf_path = f'{data_dir}/MCD15A2H.061_500m_aid0001.nc'
    surface_path = f'{data_dir}/VNP09H1.001_500m_aid0001.nc'
    surface_ds = Dataset(surface_path, mode='r')
    s_dates_df = get_monthly_latest_date(surface_ds, 's')

    leaf_ds = Dataset(leaf_path, mode='r')
    l_dates_df = get_monthly_latest_date(leaf_ds, prefix='l')

    vegetation_ds = Dataset(vegetation_path, mode='r')
    v_dates_df = get_monthly_latest_date(vegetation_ds, prefix='v')

    dates_df = pd.merge(s_dates_df, l_dates_df, on='m')
    dates_df = pd.merge(dates_df, v_dates_df, on='m')
    return dates_df
