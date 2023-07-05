
from configs.default_forest_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + f"deforest_train_cv{cfg.fold}.csv"
cfg.valid_df = cfg.data_dir + f"deforest_valid_cv{cfg.fold}.csv"
cfg.test_df = cfg.data_dir + "deforest_targets.csv"
cfg.output_dir = "../output/models/" + cfg.name


# model
cfg.model = "forest_model"
cfg.return_logits = True
cfg.backbone = "tf_efficientnet_b3_ns"
cfg.batch_size = 32
cfg.grad_accumulation = 1
cfg.drop_out = 0.0
cfg.seed = 2023

# DATASET
cfg.dataset = "forest_dataset"
cfg.classes = ["Deforestation"]
cfg.image_width = 256
cfg.image_height = 256
cfg.sources = ["ls8", "s1", "s2"]
cfg.max_rank = 8
cfg.input_day_delta = 32
cfg.source_date_dropout = 0.5
cfg.balanced = True
cfg.select_month = 0

cfg.BAND_LIMITS = {
    "ls8": {
        "SR_B1": (5000, 35000),
        "SR_B2": (5000, 35000),
        "SR_B3": (7000, 35000),
        "SR_B4": (7000, 35000), 
        "SR_B5": (12000, 35000),
        "SR_B6": (9000, 35000),
        "SR_B7": (9000, 35000),
        "ST_B10": (30000, 45000),
        "QA_PIXEL": (24000, 35000),
    },
    "s1": {
        "VH": (-30, 1),
        "VV": (-20, 1),
    },
    "s2": {
        "B1": (2000, 8000),
        "B2": (2000, 8000),
        "B3": (2000, 8000),
        "B4": (2000, 8000),
        "B5": (2000, 8000),
        "B6": (2000, 8000),
        "B7": (2000, 8000),
        "B8": (2000, 8000),
        "B8A": (2000, 8000),
        "B9": (3000, 10000),
        "B11": (2000, 8000),
        "B12": (2000, 8000),
        "QA60": (300, 1000),
    },
}

cfg.n_channels = cfg.max_rank * sum(
    [len(cfg.BAND_LIMITS[s]) for s in cfg.sources]
)


# OPTIMIZATION & SCHEDULE
cfg.epochs = 10
cfg.lr = 0.0005
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
