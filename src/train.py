import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from copy import copy
import os
from collections import defaultdict

from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score, average_precision_score

import wandb

from utils import (
    sync_across_gpus,
    set_seed,
    get_model,
    create_checkpoint,
    get_dataloader,
    get_optimizer,
    get_scheduler,
)

def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import cv2

    cv2.setNumThreads(0)
except:
    print("no cv2 installed, running without")

sys.path.append("configs")
sys.path.append("models")
sys.path.append("datasets")



def iou_score(y_true, y_pred):
    eps = 1e-6
    intersection = np.sum(y_pred * y_true)
    cardinality = np.sum(y_pred) + np.sum(y_true)
    score = intersection / (cardinality - intersection + eps)
    if np.sum(y_true) == 0:
        score = np.nan
        print("Warning: all zero target")
    return score


def run_predict(model, test_dataloader, valid_df, cfg):
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):
        batch = cfg.batch_to_device(data, cfg.device)
        output = model(batch)

        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]
        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    
    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]
    
        P = test_data["logits"].detach().sigmoid().cpu().numpy()
        M = test_data["masks"].detach().cpu().numpy()
        gt = M.flatten()
        preds = P[:, 0, :, :].flatten()
        row = {
            'bce': log_loss(gt, preds),
            'accuracy': accuracy_score(gt, preds > 0.5),
            'f1': f1_score(gt, preds > 0.5),
            'iou': iou_score(M, 1 * (P[:, 0, :, :] > 0.5)),
            'auc': roc_auc_score(gt, preds),
            'ap': average_precision_score(gt, preds)
        }
        print(row)
        wandb.log(row)

    if cfg.distributed: 
        torch.distributed.barrier()


def save_test_predictions(model, test_dataloader, test_df, cfg):
    model.eval()
    torch.set_grad_enabled(False)

    # store information for submission
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = cfg.batch_to_device(data, cfg.device)
        output = model(batch)
        
        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]
        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]

            test_pred = test_data["logits"].detach().sigmoid().cpu().numpy()
            print(test_pred.shape)
            np.save(f"{cfg.output_dir}/test_data_f{cfg.fold}_e{cfg.curr_epoch}.npy", test_pred)

    if cfg.distributed: 
        torch.distributed.barrier()


def train(cfg):
    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        print("RANK", cfg.local_rank)
        device = "cuda:%d" % cfg.local_rank
        cfg.device = device
        print("device", device)

        torch.cuda.set_device(cfg.local_rank)

        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process."
        )
        print(
            f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}."
        )
        garbage_collection_cuda()
        s=32
        torch.nn.functional.conv2d(
            torch.zeros(s, s, s, s, device=device),
            torch.zeros(s, s, s, s, device=device)
        )
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
        print("Group", cfg.group)
        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )  #
        print("seed", cfg.local_rank, cfg.seed)

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0

        device = "cuda:%d" % cfg.gpu
        cfg.device = device

    set_seed(cfg.seed)

    train_df = pd.read_csv(cfg.train_df)
    valid_df = pd.read_csv(cfg.valid_df)
    test_df = pd.read_csv(cfg.test_df)

    if cfg.local_rank == 0:
        print(train_df.shape, valid_df.shape, test_df.shape)

    val_dataset = cfg.CustomDataset(valid_df, mode="test", aug=cfg.val_aug, cfg=cfg)
    val_dataloader = get_dataloader(val_dataset, cfg, mode="test")

    test_dataset = cfg.CustomDataset(test_df, mode="test", aug=cfg.val_aug, cfg=cfg)
    test_dataloader = get_dataloader(test_dataset, cfg, mode="test")

    model = get_model(cfg)
    model.to(device)

    if cfg.distributed:
        model = NativeDDP(
            model, device_ids=[cfg.local_rank], find_unused_parameters=False
        )

    train_dataset = cfg.CustomDataset(train_df, mode="train", aug=cfg.train_aug, cfg=cfg)
    train_dataloader = get_dataloader(train_dataset, cfg, mode="train")
    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    val_score = 0
    if cfg.local_rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_user,
            config = vars(cfg)
        )

    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0:
            print("EPOCH:", epoch)
        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1
                cfg.curr_step += cfg.batch_size * cfg.world_size
                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                model.train()
                torch.set_grad_enabled(True)

                if (cfg.local_rank == 0) and (i == 1):
                    fb = f"{cfg.output_dir}/first_batch_e{epoch}.npy"
                    np.save(fb, data['input'].numpy())
                    fm = f"{cfg.output_dir}/first_mask_e{epoch}.npy"
                    np.save(fb, data['mask'].numpy())

                batch = cfg.batch_to_device(data, device)
                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)
                loss = output_dict["loss"]
                losses.append(loss.item())
                if cfg.grad_accumulation != 0:
                    loss /= cfg.grad_accumulation
                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.distributed:
                    torch.cuda.synchronize()
                if scheduler is not None:
                    scheduler.step()
                if cfg.local_rank == 0 and cfg.curr_step % (cfg.batch_size) == 0:
                    wandb.log({"train_loss": np.mean(losses[-10:])})
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]})
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

            print(f"Mean train_loss {np.mean(losses):.4f}")

        if cfg.distributed:
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(
                    cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
                )
                torch.save(
                    checkpoint,
                    f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth",
                )
        
        if cfg.test:
            run_predict(model, val_dataloader, valid_df, cfg)
            save_test_predictions(model, test_dataloader, test_df, cfg)

    if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
        checkpoint = create_checkpoint(
            cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
        )
        torch.save(
            checkpoint,
            f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth",
        )
    return val_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser_args, other_args = parser.parse_known_args(sys.argv)

    fold = parser_args.fold

    cfg = copy(importlib.import_module(parser_args.config).cfg)

    cfg.fold = fold

    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

    os.makedirs(str(cfg.output_dir), exist_ok=True)
    result = train(cfg)
