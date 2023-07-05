
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
cfg.backbone = "tf_efficientnet_b5_ns"
cfg.batch_size = 32
cfg.grad_accumulation = 1
cfg.drop_out = 0.0
cfg.seed = 2020

# DATASET
cfg.dataset = "forest_dataset"
cfg.classes = ["Deforestation"]
cfg.image_width = 256
cfg.image_height = 256
cfg.sources = ["ls8", "s1", "s2"]
cfg.max_rank = 1
cfg.input_day_delta = 30
cfg.source_date_dropout = 0.0
cfg.balanced = True
cfg.select_month = 0

S2_MIN = 300
S2_MAX = 3000
LS8_MIN = 3000
LS8_MAX = 40000
cfg.BAND_LIMITS = {
    "ls8": {
        # "SR_B1": (LS8_MIN, LS8_MAX),
        # "SR_B2": (LS8_MIN, LS8_MAX),
        "SR_B3": (LS8_MIN, LS8_MAX),
        "SR_B4": (LS8_MIN, LS8_MAX), 
        "SR_B5": (LS8_MIN, LS8_MAX),
        "SR_B6": (LS8_MIN, LS8_MAX),
        "SR_B7": (LS8_MIN, LS8_MAX),
        # "ST_B10": (LS8_MIN, LS8_MAX),
        "QA_PIXEL": (24000, 30000),
    },
    "s1": {
        "VH": (-30, 3),
        "VV": (-30, 3),
    },
    "s2": {
        # "B1": (S2_MIN, S2_MAX),
        # "B2": (S2_MIN, S2_MAX),
        "B3": (S2_MIN, S2_MAX),
        "B4": (S2_MIN, S2_MAX),
        # "B5": (S2_MIN, S2_MAX),
        # "B6": (S2_MIN, S2_MAX),
        "B7": (S2_MIN, S2_MAX),
        "B8": (S2_MIN, S2_MAX),
        # "B8A": (S2_MIN, S2_MAX),
        # "B9": (S2_MIN, S2_MAX),
        "B11": (S2_MIN, S2_MAX),
        "B12": (S2_MIN, S2_MAX),
        "QA60": (300, 1000),
    },
}

cfg.n_channels = cfg.max_rank * sum(
    [len(cfg.BAND_LIMITS[s]) for s in cfg.sources]
)


# OPTIMIZATION & SCHEDULE
cfg.epochs = 10
cfg.lr = 0.0007
cfg.optimizer = "Adam"
cfg.warmup = 0

# AUGMENTATION
cfg.train_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.VerticalFlip(always_apply=False, p=0.5),
    A.RandomRotate90(always_apply=False, p=0.5),
    A.Transpose(always_apply=False, p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.0625, rotate_limit=10, p=0.5,
    ),
    A.PixelDropout(dropout_prob=0.01, per_channel=True, p=0.5, drop_value=1.0)
    
])

cfg.val_aug = A.Compose([
    A.Resize(height=cfg.image_height, width=cfg.image_width, always_apply=True, p=1),
])
