from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW, Adam
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler
from torch import nn, optim
import importlib
import math


class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        print("TOTAL SIZE", self.total_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples
        ]
        print(
            "SAMPLES",
            self.rank * self.num_samples,
            self.rank * self.num_samples + self.num_samples,
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)


def set_seed(seed=1987):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)   
    return net


def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):
    state_dict = model.state_dict()
    if cfg.save_weights_only:
        checkpoint = {"model": state_dict}
        return checkpoint
    
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def get_dataloader(ds, cfg, mode='train'):

    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='test':
        dl = get_test_dataloader(ds, cfg)
    else:
        pass
    return dl



def get_train_dataloader(train_ds, cfg):

    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed
        )
    else:
        try:
            if cfg.random_sampler_frac > 0:
                num_samples = int(len(train_ds) * cfg.random_sampler_frac)
                sample_weights = train_ds.sample_weights
                sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples)
            else:
                sampler = None
        except:
            sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_test_dataloader(test_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            test_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(test_ds)

    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def get_optimizer(model, cfg):
    params = model.parameters()

    if cfg.optimizer == "AdamW_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = AdamW(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adam_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = Adam(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adam":
        optimizer = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    if cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
            num_cycles = cfg.num_cycles,
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
    else:
        scheduler = None

    return scheduler

