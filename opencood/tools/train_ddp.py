# -*- coding: utf-8 -*-
# DDP training version for Where2comm / E-3DSNN

import argparse
import os
import statistics

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.loss_utils import get_loss_mode, compute_training_loss

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="DDP training")

    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="yaml configuration file"
    )

    parser.add_argument(
        "--model_dir",
        default="",
        help="continued training path"
    )

    parser.add_argument(
        "--pretrained_path",
        default="",
        help="Checkpoint file or log dir used only for model weight initialization"
    )

    parser.add_argument(
        "--fusion_method",
        "-f",
        default="intermediate",
        help="passed to inference"
    )

    opt = parser.parse_args()
    return opt


def main():

    opt = train_parser()

    if opt.model_dir and opt.pretrained_path:
        raise ValueError(
            "--model_dir resumes an existing experiment, while "
            "--pretrained_path initializes a new experiment. Use only one."
        )

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # ============================================================
    # 1. DDP initialization
    # ============================================================

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:

        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        device = torch.device(
            "cuda",
            local_rank
        )

    else:

        local_rank = 0
        rank = 0
        world_size = 1

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    is_main_process = (rank == 0)

    def rank_log(message):
        print(
            "[rank {}] {}".format(rank, message),
            flush=True
        )

    if is_main_process:
        print("========== DDP ==========")
        print("Distributed :", distributed)
        print("World size  :", world_size)
        print("Rank        :", rank)
        print("Local rank  :", local_rank)
        print("Device      :", device)
        print("=========================")

    # ============================================================
    # 2. Dataset
    # ============================================================

    if is_main_process:
        print("Dataset Building")

    opencood_train_dataset = build_dataset(
        hypes,
        visualize=False,
        train=True
    )

    # Build validation dataset on every rank.
    # Each rank will evaluate a disjoint subset.
    opencood_validate_dataset = build_dataset(
        hypes,
        visualize=False,
        train=False
    )

    rank_log(
        "validation dataset ready: {}".format(
            len(opencood_validate_dataset)
        )
    )

    # ============================================================
    # 3. Distributed sampler
    # ============================================================

    if distributed:
        train_sampler = DistributedSampler(
            opencood_train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    else:
        train_sampler = None

    num_workers = hypes["train_params"].get(
        "num_workers",
        4
    )
    persistent_workers = bool(
        hypes["train_params"].get("persistent_workers", False)
    )
    prefetch_factor = int(
        hypes["train_params"].get("prefetch_factor", 2)
    )
    empty_cache_each_iter = bool(
        hypes["train_params"].get("empty_cache_each_iter", True)
    )

    # Independent worker settings for training and validation.
    train_num_workers = int(
        hypes["train_params"].get(
            "train_num_workers", num_workers
        )
    )
    val_num_workers = int(
        hypes["train_params"].get("val_num_workers", 0)
    )

    train_persistent_workers = bool(
        hypes["train_params"].get(
            "train_persistent_workers", persistent_workers
        )
    )
    val_persistent_workers = bool(
        hypes["train_params"].get(
            "val_persistent_workers", False
        )
    )

    if train_num_workers < 0 or val_num_workers < 0:
        raise ValueError("DataLoader workers must be >= 0")

    if is_main_process:
        print(
            "[DataLoader Split] train_workers={} "
            "val_workers={} train_persistent={} "
            "val_persistent={} prefetch={}".format(
                train_num_workers,
                val_num_workers,
                train_persistent_workers,
                val_persistent_workers,
                prefetch_factor
            )
        )
    if num_workers < 0:
        raise ValueError("train_params.num_workers must be >= 0")
    if prefetch_factor < 1:
        raise ValueError("train_params.prefetch_factor must be >= 1")
    if is_main_process:
        print(
            "[DataLoader] num_workers={} persistent_workers={} "
            "prefetch_factor={} empty_cache_each_iter={}".format(
                num_workers, persistent_workers and num_workers > 0,
                prefetch_factor if num_workers > 0 else None,
                empty_cache_each_iter
            )
        )

    train_loader = DataLoader(
        opencood_train_dataset,

        # IMPORTANT:
        # batch_size is PER GPU under DDP.
        batch_size=hypes["train_params"]["batch_size"],

        num_workers=train_num_workers,

        persistent_workers=(train_persistent_workers and train_num_workers > 0),

        prefetch_factor=(prefetch_factor if train_num_workers > 0 else 2),
        collate_fn=(
            opencood_train_dataset.collate_batch_train
        ),

        shuffle=(train_sampler is None),
        sampler=train_sampler,

        pin_memory=True,
        drop_last=True
    )

    # Distributed validation: disjoint samples per rank.
    if distributed:
        val_indices = list(
            range(rank, len(opencood_validate_dataset), world_size)
        )
        val_dataset = Subset(
            opencood_validate_dataset,
            val_indices
        )
    else:
        val_dataset = opencood_validate_dataset

    val_loader = DataLoader(
        val_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=val_num_workers,
        persistent_workers=(
            val_persistent_workers and val_num_workers > 0
        ),
        prefetch_factor=(
            prefetch_factor if val_num_workers > 0 else 2
        ),
        collate_fn=(
            opencood_validate_dataset.collate_batch_train
        ),
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    rank_log(
        "validation shard: {} samples, {} batches".format(
            len(val_dataset),
            len(val_loader)
        )
    )

    # ============================================================
    # 4. Model
    # ============================================================

    if is_main_process:
        print("Creating Model")

    model = train_utils.create_model(hypes)
    model.to(device)

    # ============================================================
    # 5. Resume / output directory
    # ============================================================

    if opt.model_dir:

        saved_path = opt.model_dir

        init_epoch, model = train_utils.load_saved_model(
            saved_path,
            model
        )

    else:

        init_epoch = 0

        # Only rank 0 creates the result directory.
        if is_main_process:
            saved_path = train_utils.setup_train(hypes)
        else:
            saved_path = None

        # Broadcast saved_path to other ranks.
        if distributed:
            saved_path_list = [saved_path]

            dist.broadcast_object_list(
                saved_path_list,
                src=0
            )

            saved_path = saved_path_list[0]

        # Pretrained initialization is different from resume:
        # load model weights only, keep a fresh epoch/optimizer/scheduler.
        if opt.pretrained_path:
            model = train_utils.load_pretrained_model(
                opt.pretrained_path,
                model,
                allowed_missing_prefixes=(
                    "fusion_net.dfa.",
                    "backbone.multi_scale_neck.",
                ),
                allow_unexpected=False,
            )

    # Parameter audit before DDP wrapping.
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        trainable_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        print("Total parameters      :", total_params)
        print("Trainable parameters  :", trainable_params)
        print("Trainable tensors      :", len(trainable_names))
        print("Trainable parameter names:")
        for name in trainable_names:
            print("  " + name)

    # ============================================================
    # 6. DDP wrapper
    # ============================================================

    if distributed:

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,

            # Where2comm contains multiple auxiliary branches.
            # Use True first for robustness.
            find_unused_parameters=True
        )

    # ============================================================
    # 7. Loss / optimizer / scheduler
    # ============================================================

    criterion = train_utils.create_loss(hypes)
    loss_mode = get_loss_mode(hypes)

    if is_main_process:
        print("Supervision mode :", loss_mode)

    optimizer = train_utils.setup_optimizer(
        hypes,
        model
    )

    if opt.model_dir:

        scheduler = train_utils.setup_lr_schedular(
            hypes,
            optimizer,
            init_epoch=init_epoch
        )

    else:

        scheduler = train_utils.setup_lr_schedular(
            hypes,
            optimizer
        )

    # ============================================================
    # 8. TensorBoard: rank 0 only
    # ============================================================

    if is_main_process:
        writer = SummaryWriter(saved_path)
    else:
        writer = None

    if is_main_process:

        print("Training start")

        print(
            "Train samples        :",
            len(opencood_train_dataset)
        )

        print(
            "Batch size / GPU     :",
            hypes["train_params"]["batch_size"]
        )

        print(
            "World size           :",
            world_size
        )

        print(
            "Global batch size    :",
            hypes["train_params"]["batch_size"]
            * world_size
        )

        print(
            "Iterations / epoch   :",
            len(train_loader)
        )

    epoches = hypes["train_params"]["epoches"]

    # ============================================================
    # 9. Training loop
    # ============================================================

    for epoch in range(
        init_epoch,
        max(epoches, init_epoch)
    ):

        # Required for proper DDP shuffle.
        if distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process:

            for param_group in optimizer.param_groups:
                print(
                    "learning rate %f"
                    % param_group["lr"]
                )

        for i, batch_data in enumerate(train_loader):

            debug_step = (
                i < 2
                or (i + 1) % 100 == 0
                or (i + 1) >= len(train_loader) - 2
            )

            if debug_step:
                rank_log(
                    "epoch={} step={}/{} batch_fetched={}".format(
                        epoch,
                        i + 1,
                        len(train_loader),
                        batch_data is not None
                    )
                )

            # ----------------------------------------------------
            # IMPORTANT:
            # if one rank gets None, all ranks skip the step.
            # Otherwise DDP can deadlock during backward.
            # ----------------------------------------------------

            if distributed:

                local_valid = torch.tensor(
                    0 if batch_data is None else 1,
                    device=device,
                    dtype=torch.int32
                )

                if debug_step:
                    rank_log(
                        "step={} entering local_valid all_reduce".format(
                            i + 1
                        )
                    )

                dist.all_reduce(
                    local_valid,
                    op=dist.ReduceOp.MIN
                )

                if debug_step:
                    rank_log(
                        "step={} completed local_valid all_reduce".format(
                            i + 1
                        )
                    )

                if local_valid.item() == 0:
                    if debug_step:
                        rank_log(
                            "step={} skipped: invalid batch on at least one rank".format(
                                i + 1
                            )
                        )
                    continue

            else:

                if batch_data is None:
                    continue

            if debug_step:
                rank_log(
                    "step={} entering forward".format(i + 1)
                )

            model.train()

            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(
                batch_data,
                device
            )

            batch_data["ego"]["epoch"] = epoch

            # ----------------------------------------------------
            # Forward
            # ----------------------------------------------------

            output_dict = model(
                batch_data["ego"]
            )

            if debug_step:
                rank_log(
                    "step={} forward completed".format(i + 1)
                )

            # ----------------------------------------------------
            # Configurable supervision objective (shared with train.py)
            # ----------------------------------------------------

            final_loss = compute_training_loss(
                output_dict,
                batch_data,
                hypes,
                criterion,
                loss_mode
            )

            if debug_step:
                rank_log(
                    "step={} loss computed: {:.6f}".format(
                        i + 1,
                        float(final_loss.detach().item())
                    )
                )

            # ----------------------------------------------------
            # Log: rank 0 only
            # ----------------------------------------------------

            if is_main_process:

                criterion.logging(
                    epoch,
                    i,
                    len(train_loader),
                    writer
                )

            # ----------------------------------------------------
            # Backward / synchronized DDP gradients
            # ----------------------------------------------------

            if debug_step:
                rank_log(
                    "step={} entering backward".format(i + 1)
                )

            final_loss.backward()

            if debug_step:
                rank_log(
                    "step={} backward completed".format(i + 1)
                )

            optimizer.step()

            if debug_step:
                rank_log(
                    "step={} optimizer completed".format(i + 1)
                )

            del batch_data
            del output_dict
            del final_loss

            if empty_cache_each_iter:
                torch.cuda.empty_cache()

        # ========================================================
        # 10. Epoch completion
        # ========================================================

        rank_log(
            "epoch {} training loop completed".format(epoch)
        )

        # ========================================================
        # 11. Checkpoint FIRST
        # Preserve completed training weights before validation.
        # ========================================================

        if (
            is_main_process
            and
            epoch % hypes["train_params"]["save_freq"] == 0
        ):

            state_dict = (
                model.module.state_dict()
                if distributed
                else model.state_dict()
            )

            ckpt_path = os.path.join(
                saved_path,
                "net_epoch%d.pth" % (epoch + 1)
            )

            torch.save(
                state_dict,
                ckpt_path
            )

            rank_log(
                "checkpoint saved: {} ({} bytes)".format(
                    ckpt_path,
                    os.path.getsize(ckpt_path)
                )
            )

        # All ranks wait until rank 0 finishes checkpoint saving.
        if distributed:
            rank_log(
                "epoch {} entering checkpoint barrier".format(epoch)
            )
            dist.barrier()
            rank_log(
                "epoch {} passed checkpoint barrier".format(epoch)
            )

        # ========================================================
        # 12. Distributed validation
        # Validation frequency: every N completed epochs.
        # ========================================================

        eval_freq = int(
            hypes["train_params"]["eval_freq"]
        )

        if eval_freq > 0 and (epoch + 1) % eval_freq == 0:

            eval_model = (
                model.module
                if distributed
                else model
            )

            eval_model.eval()

            valid_loss_sum = 0.0
            valid_sample_count = 0

            rank_log(
                "epoch {} validation started".format(epoch + 1)
            )

            with torch.no_grad():

                for i, batch_data in enumerate(val_loader):

                    if batch_data is None:
                        continue

                    batch_data = train_utils.to_device(
                        batch_data,
                        device
                    )

                    batch_data["ego"]["epoch"] = epoch

                    output_dict = eval_model(
                        batch_data["ego"]
                    )

                    final_loss = compute_training_loss(
                        output_dict,
                        batch_data,
                        hypes,
                        criterion,
                        loss_mode
                    )

                    sample_count = int(
                        batch_data["ego"]["record_len"].shape[0]
                    )

                    valid_loss_sum += (
                        float(final_loss.detach().item())
                        * sample_count
                    )

                    valid_sample_count += sample_count

                    if (
                        i == 0
                        or (i + 1) % 100 == 0
                        or (i + 1) == len(val_loader)
                    ):
                        rank_log(
                            "validation step={}/{} samples={}".format(
                                i + 1,
                                len(val_loader),
                                valid_sample_count
                            )
                        )

                    del batch_data
                    del output_dict
                    del final_loss

            # Aggregate weighted validation loss across ranks.
            if distributed:

                valid_stats = torch.tensor(
                    [
                        valid_loss_sum,
                        float(valid_sample_count)
                    ],
                    dtype=torch.float64,
                    device=device
                )

                rank_log(
                    "entering validation metric all_reduce"
                )

                dist.all_reduce(
                    valid_stats,
                    op=dist.ReduceOp.SUM
                )

                rank_log(
                    "completed validation metric all_reduce"
                )

                valid_loss_sum = float(
                    valid_stats[0].item()
                )

                valid_sample_count = int(
                    valid_stats[1].item()
                )

            if valid_sample_count > 0:

                valid_ave_loss = (
                    valid_loss_sum / valid_sample_count
                )

                if is_main_process:

                    print(
                        "At epoch %d, the validation loss is %f"
                        % (
                            epoch,
                            valid_ave_loss
                        ),
                        flush=True
                    )

                    writer.add_scalar(
                        "Validate_Loss",
                        valid_ave_loss,
                        epoch
                    )

            else:

                rank_log(
                    "WARNING: validation produced zero valid samples"
                )

            eval_model.train()

            if empty_cache_each_iter:
                torch.cuda.empty_cache()

            rank_log(
                "epoch {} validation completed".format(epoch + 1)
            )

        # ========================================================
        # 13. LR scheduler: every rank MUST execute this.
        # ========================================================

        scheduler.step(epoch)

        rank_log(
            "epoch {} scheduler completed".format(epoch + 1)
        )

    # ============================================================
    # 13. Finish
    # ============================================================

    if distributed:
        dist.barrier()

    if is_main_process:

        print(
            "Training Finished, "
            "checkpoints saved to %s"
            % saved_path
        )

        if writer is not None:
            writer.close()

    torch.cuda.empty_cache()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
