import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
import timm
import numpy as np

from segmentation_models_pytorch.decoders.unet.model import (
    UnetDecoder,
    SegmentationHead,
)


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.n_classes = 1
        in_chans = cfg.n_channels

        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            in_chans=in_chans,
        )
        encoder_channels = tuple(
            [in_chans]
            + [
                self.encoder.feature_info[i]["num_chs"]
                for i in range(len(self.encoder.feature_info))
            ]
        )
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),  # 256??
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=3,
        )

        self.bce_seg = nn.BCEWithLogitsLoss()

        self.return_logits = cfg.return_logits

    def forward(self, batch):

        x_in = batch["input"]

        enc_out = self.encoder(x_in)

        decoder_out = self.decoder(*[x_in] + enc_out)
        x_seg = self.segmentation_head(decoder_out)

        output = {}
        if (not self.training) & self.return_logits:
            output["logits"] = x_seg
            output["masks"] = batch["mask"]

        if self.training:
    
            one_hot_mask = batch["mask"][:, None]
            seg_loss = self.bce_seg(x_seg, one_hot_mask.float())
            loss = seg_loss

            output["loss"] = loss
            output["seg_loss"] = seg_loss

        return output
