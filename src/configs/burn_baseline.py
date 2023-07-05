
from configs.default_forest_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + f"fire_old_train_cv{cfg.fold}.csv"
cfg.valid_df = cfg.data_dir + f"fire_old_valid_cv{cfg.fold}.csv"
cfg.test_df = cfg.data_dir + "fire_old_targets.csv"
cfg.output_dir = "../output/models/" + cfg.name


# model
cfg.model = "forest_model"
cfg.return_logits = True
# cfg.backbone = "resnet50"
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.batch_size = 32
cfg.grad_accumulation = 1
cfg.drop_out = 0.0
cfg.seed = 1987

# DATASET
cfg.dataset = "burn_dataset"
cfg.classes = ["Burn"]
cfg.image_size = 128
cfg.image_width = cfg.image_size
cfg.image_height = cfg.image_size
cfg.max_rank = 10
cfg.input_day_delta = 91
cfg.source_date_dropout = 0.5
cfg.source_date_keep_ratio = 0.8
cfg.balanced = True
cfg.sources = ["ls5"]
cfg.BAND_LIMITS = {
    "ls5": {
        "QA_PIXEL": (5500, 7500),
        "SR_B1": (5000, 50000),
        "SR_B2": (5000, 50000),
        "SR_B3": (5000, 50000),
        "SR_B4": (5000, 50000), 
        "SR_B5": (5000, 50000),
        "SR_B7": (5000, 50000),
        "ST_B6": (0, 50000),
    },
}

cfg.n_channels = cfg.max_rank * sum(
    [len(cfg.BAND_LIMITS[s]) for s in cfg.sources]
)

# OPTIMIZATION & SCHEDULE
cfg.epochs = 20
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
