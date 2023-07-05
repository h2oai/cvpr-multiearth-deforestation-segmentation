
from configs.default_forest_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "tr0.csv"
cfg.valid_df = cfg.data_dir + "tr1.csv"
cfg.test_df = cfg.data_dir + "tr1.csv"
cfg.output_dir = "../output/models/" + cfg.name


# model
cfg.model = "forest_model"
cfg.return_logits = True
cfg.backbone = "resnet50"
cfg.batch_size = 32
cfg.grad_accumulation = 1
cfg.drop_out = 0.0
cfg.seed = 1987

# DATASET
cfg.dataset = "forest_dataset"
cfg.classes = ["Deforestation"]
cfg.image_width = 256
cfg.image_height = 256
cfg.max_rank = 2

cfg.BAND_LIMITS = {
    "ls8": {
#         "SR_B1": (5000, 45000),
        "SR_B2": (5000, 45000),
        "SR_B3": (5000, 45000),
        "SR_B4": (5000, 45000), 
        "SR_B5": (5000, 45000),
        "SR_B6": (5000, 45000),
        "SR_B7": (5000, 45000),
#         "ST_B10": (5000, 45000),
    },
    "s1": {
        "VH": (-30, 2),
        "VV": (-20, 2),
    },
    "s2": {
#         "B1": (1000, 10000),
        "B2": (1000, 10000),
        "B3": (1000, 10000),
        "B4": (1000, 10000),
        "B5": (1000, 10000),
        "B6": (1000, 10000),
        "B7": (1000, 10000),
        "B8": (1000, 10000),
#         "B8A": (1000, 10000),
#         "B9": (1000, 10000),
        "B11": (1000, 10000),
        "B12": (1000, 10000),
    },
}

cfg.n_channels = cfg.max_rank * sum(
    [len(cfg.BAND_LIMITS[s]) for s in cfg.sources]
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
