from default_config import basic_cfg

cfg = basic_cfg

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pool = "avg"

cfg.normalization = "minmax"

cfg.image_width = 256
cfg.image_height = 256

cfg.wandb_project = "fire"

basic_cv_cfg = cfg
