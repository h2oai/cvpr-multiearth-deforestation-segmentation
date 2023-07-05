import os

from multiprocessing import Pool
from tqdm import tqdm

from config import IMAGE_SIZE, LAST_YEAR, REGIONS
from firms_tools import create_tile_images


def update(res):
    pbar.update()


def error(e):
    print(e)


if __name__ == "__main__":

    pool = Pool(processes=2)
    pbar = tqdm(total=len(os.listdir('maps')))

    ds_name = f"fs_{IMAGE_SIZE}"
    for region in REGIONS:
        os.makedirs(f"data/{ds_name}/{region}", exist_ok=True)
        for y in range(2013, LAST_YEAR + 1):
            for m in range(1, 13):
                pool.apply_async(
                    create_tile_images,
                    args=(ds_name, region, y, m),
                    callback=update,
                    error_callback=error,
                )

    pool.close()
    pool.join()
    pbar.close()
