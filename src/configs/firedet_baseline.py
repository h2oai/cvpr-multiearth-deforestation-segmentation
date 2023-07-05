
from configs.default_fire_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "fire_train_sample_T0_wo_roi.csv"
cfg.valid_df = cfg.data_dir + "fire_valid_roi.csv"
cfg.test_df = cfg.data_dir + "fire_valid_roi.csv"
cfg.output_dir = "../output/models/" + cfg.name


# model
cfg.model = "forest_model"
cfg.return_logits = True
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.batch_size = 32
cfg.grad_accumulation = 1
cfg.drop_out = 0.0
cfg.seed = 1987

# DATASET
cfg.dataset = "fire_dataset"
cfg.classes = ["Burn"]
cfg.image_width = 256
cfg.image_height = 256

# cfg.filter_i_range = (5000, 15000)  # Speed up training & validation
cfg.filter_i_range = None

cfg.target_lag = 0  # predict current month

cfg.layer_windows = {
    "s1": (0, 3),
    "s2": (0, 3),
    "s3": (0, 3),
    "l1": (0, 3),
    "l2": (0, 3),
    "v1": (0, 3),
    "v2": (0, 3),
    "v3": (0, 3),
    "v4": (0, 3),
    "v5": (0, 3),
    "v6": (0, 3),
    "v7": (0, 3),
    "firms_cnt": (0, 1),
    "firms_seasonal": (0, 1),
    "firms_recency": (0, 1),
    "firms_current": (0, 4),
    "burn": (1, 4),  # can not use current burn as a feature
}

cfg.n_channels = sum(
    [(b - a) for _, (a, b) in cfg.layer_windows.items()]
)


# OPTIMIZATION & SCHEDULE
cfg.epochs = 5
cfg.lr = 0.001
cfg.optimizer = "AdamW"
cfg.warmup = 0

# AUGMENTATION
cfg.train_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.VerticalFlip(always_apply=False, p=0.5),
    A.RandomRotate90(always_apply=False, p=0.5),
    
])

cfg.val_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
])
