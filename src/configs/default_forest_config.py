from default_config import basic_cfg

cfg = basic_cfg

# img model
cfg.sources = ["ls8", "s1", "s2"]
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pool = "avg"

cfg.normalization = "minmax"

cfg.image_width = 256
cfg.image_height = 256
cfg.input_day_delta = 0
cfg.balanced = False
cfg.wandb_project = "forest"
cfg.source_date_dropout = 0.0

basic_cv_cfg = cfg
