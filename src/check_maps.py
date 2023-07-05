import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import argparse
import cv2

from config import IMAGE_SIZE, SOUTH_AMERICA, FIRECCI_SHAPE


def update(res):
    pbar.update()


def error(e):
    print(e)



def check_example_maps(f):
    M = np.load(f"maps/{f}")
    name = f.split(".")[0]

    if os.path.exists(f"example_images/{name}.png"):
        return

    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 10), sharex=True, sharey=True
    )
    for i, layer in enumerate(["s1", "s2", "l1", "l2", "v1", "v2"]):
        ax = axs[i // ncols, i % ncols]
        X = M[layer]
        _ = ax.imshow(X)
        _ = ax.set_title(layer)

    _ = plt.suptitle(name)
    _ = plt.tight_layout()
    _ = fig.savefig(f"example_images/{name}.png")
    plt.close(fig)

    del fig, axs
    del M.f, X
    M.close()
    gc.collect()


def create_tile_features(f):
    M = np.load(f"maps/{f}")
    region, y, m = f.split(".")[0].split("-")
    y, m = int(y), int(m)
    result = []
    feature_filename = f"features/sf-{f.replace('npz', 'csv')}"

    if os.path.exists(feature_filename):
        return

    n_lats, n_lons = FIRECCI_SHAPE[SOUTH_AMERICA]
    layers_to_check = ["s1", "s2", "s3", "l1", "l2", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    for layer in layers_to_check:
        # resize to FIRECCI resulution
        X = cv2.resize(M[layer], (n_lons, n_lats))
        for i in range(0, X.shape[0], IMAGE_SIZE):
            for j in range(0, X.shape[1], IMAGE_SIZE):
                tile = X[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE]
                n_positive_pixels = np.sum(tile > 0)
                features = [
                    region,
                    y,
                    m,
                    i,
                    j,
                    layer,
                    n_positive_pixels,
                    tile.shape[0],
                    tile.shape[1],
                ]
                if n_positive_pixels == 0:
                    features += [0] * 7
                else:
                    x = tile[tile > 0]
                    features += [
                        np.percentile(x, 10),
                        np.percentile(x, 20),
                        np.percentile(x, 50),
                        np.percentile(x, 80),
                        np.percentile(x, 90),
                        x.mean(),
                        x.std(),
                    ]
                result.append(features)
    cols = [
        "region",
        "y",
        "m",
        "i",
        "j",
        "layer",
        "n_positive_pixels",
        "s0",
        "s1",
        "x_p10",
        "x_p20",
        "x_p50",
        "x_p80",
        "x_p90",
        "x_mean",
        "x_std",
    ]
    result_df = pd.DataFrame(result, columns=cols)
    result_df.to_csv(feature_filename, index=False)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", type=str, default="check-maps", help="check-maps or create-features")
    args = vars(ap.parse_args())
    print(args)
    
    os.makedirs("features", exist_ok=True)
    os.makedirs("example_images", exist_ok=True)

    N_WORKERS = 4 if args["task"] == "check-maps" else 24

    pool = Pool(processes=N_WORKERS)
    map_files = [f for f in os.listdir("maps") if f.startswith(SOUTH_AMERICA)]
    print(len(map_files))
    pbar = tqdm(total=len(map_files))

    if args["task"] == "create-features":
        for map_file in map_files:
            result = pool.apply_async(
                create_tile_features,
                args=(map_file,),
                callback=update,
                error_callback=error,
            )
    elif args["task"] == "check-maps":
        for map_file in map_files:
            result = pool.apply_async(
                check_example_maps,
                args=(map_file,),
                callback=update,
                error_callback=error,
            )

    pool.close()
    pool.join()
    pbar.close()

