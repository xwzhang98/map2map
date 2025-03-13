import os
import socket
import sys
import time
from pprint import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import models
from .data import FieldDataset, DistFieldSampler
from .models import (
    narrow_cast,
    resample,
    lag2eul,
    wgan_grad_penalty,
)
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power

ckpt_link = "checkpoint.pt"


def node_worker(args):
    if "SLURM_STEP_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_STEP_NUM_NODES"])
    elif "SLURM_JOB_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    else:
        raise KeyError("missing node counts in slurm env")
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node

    node = int(os.environ["SLURM_NODEID"])

    if args.gpus_per_node < 1:
        raise RuntimeError("GPU not found on node {}".format(node))

    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    device = torch.device("cuda", 0)

    rank = args.gpus_per_node * node + local_rank

    # Need randomness across processes, for sampler, augmentation, noise etc.
    # Note DDP broadcasts initial model states from rank 0
    torch.manual_seed(args.seed + rank)
    # good practice to disable cudnn.benchmark if enabling cudnn.deterministic
    # torch.backends.cudnn.deterministic = True

    dist_init(rank, args)

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        style_pattern=args.train_style_pattern,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )

    train_sampler = DistFieldSampler(
        train_dataset,
        shuffle=True,
        div_data=args.div_data,
        div_shuffle_dist=args.div_shuffle_dist,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan
    args.style_size = train_dataset.style_size

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(
        sum(args.in_chan),
        sum(args.out_chan),
        style_size=args.style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], process_group=dist.new_group()
    )

    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **args.scheduler_args)

    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    if args.adv:
        adv_model = import_attr(args.adv_model, models, callback_at=args.callback_at)
        adv_model = adv_model(
            sum(args.in_chan + args.out_chan) if args.cgan else sum(args.out_chan),
            1,
            style_size=args.style_size,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        adv_model.to(device)
        adv_model = DistributedDataParallel(
            adv_model,
            device_ids=[device],
            process_group=dist.new_group(),
        )

        adv_criterion = import_attr(
            args.adv_criterion, nn, models, callback_at=args.callback_at
        )
        adv_criterion = adv_criterion()
        adv_criterion.to(device)

        adv_optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
        adv_optimizer = adv_optimizer(
            adv_model.parameters(),
            lr=args.adv_lr,
            **args.adv_optimizer_args,
        )
        adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            adv_optimizer, **args.scheduler_args
        )

    if (
        args.load_state == ckpt_link
        and not os.path.isfile(ckpt_link)
        or not args.load_state
    ):
        if args.init_weight_std is not None:
            model.apply(init_weights)

            if args.adv:
                adv_model.apply(init_weights)

        start_epoch = 0

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state["epoch"]

        load_model_state_dict(
            model.module, state["model"], strict=args.load_state_strict
        )

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])

        if args.adv:
            if "adv_model" in state:
                load_model_state_dict(
                    adv_model.module, state["adv_model"], strict=args.load_state_strict
                )

            if "adv_optimizer" in state:
                adv_optimizer.load_state_dict(state["adv_optimizer"])
            if "adv_scheduler" in state:
                adv_scheduler.load_state_dict(state["adv_scheduler"])

        torch.set_rng_state(state["rng"].cpu())  # move rng state back

        if rank == 0:
            min_loss = state["min_loss"]
            if args.adv and "adv_model" not in state:
                min_loss = None  # restarting with adversary wipes the record

            print(
                "state at epoch {} loaded from {}".format(
                    state["epoch"], args.load_state
                ),
                flush=True,
            )

        del state

    torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        print("pytorch {}".format(torch.__version__))
        pprint(vars(args))
        sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            adv_model,
            adv_criterion,
            adv_optimizer,
            adv_scheduler,
            logger,
            device,
            args,
        )
        epoch_loss = train_loss

        if rank == 0:
            logger.flush()

            if (
                min_loss is None or epoch_loss[0] < min_loss[0]
            ) and epoch >= args.adv_start:
                min_loss = epoch_loss

            state = {
                "epoch": epoch + 1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "min_loss": min_loss,
            }
            if args.adv:
                state.update(
                    {
                        "adv_model": adv_model.module.state_dict(),
                        "adv_optimizer": adv_optimizer.state_dict(),
                        "adv_scheduler": adv_scheduler.state_dict(),
                    }
                )

            state_file = "state_{}.pt".format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = "{}.pt".format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()


def train(
    epoch,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    adv_model,
    adv_criterion,
    adv_optimizer,
    adv_scheduler,
    logger,
    device,
    args,
):
    EUL_SCALE_FACTOR = 2

    model.train()
    if args.adv:
        adv_model.train()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.log_interval <= args.adv_wgan_gp_interval or args.adv_wgan_gp_interval < 1:
        adv_wgan_gp_log_interval = args.log_interval
    else:
        adv_wgan_gp_log_interval = (
            args.log_interval // args.adv_wgan_gp_interval * args.adv_wgan_gp_interval
        )

    # loss, loss_adv, adv_loss, adv_loss_fake, adv_loss_real
    # loss: generator (model) supervised loss
    # loss_adv: generator (model) adversarial loss
    # adv_loss: discriminator (adv_model) loss
    epoch_loss = torch.zeros(5, dtype=torch.float32, device=device)
    fake = torch.zeros([1], dtype=torch.float32, device=device)
    real = torch.ones([1], dtype=torch.float32, device=device)
    adv_real = torch.full(
        [1], args.adv_label_smoothing, dtype=torch.float32, device=device
    )

    for i, data in enumerate(loader):
        batch = epoch * len(loader) + i + 1

        input, target, style = data["input"], data["target"], data["style"]

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        style = style.to(device, non_blocking=True)

        output = model(input, style)
        if batch <= 5 or (i + 1) % 200 == 0 and rank == 0:
            print("##### batch :", batch)
            print("input shape :", input.shape)
            print("output shape :", output.shape)
            print("target shape :", target.shape)
            print("style shape :", style.shape)

        if hasattr(model.module, "scale_factor") and model.module.scale_factor != 1:
            input = resample(input, model.module.scale_factor, narrow=False)
        input, output, target = narrow_cast(input, output, target)
        if batch <= 5 and rank == 0:
            print("narrowed shape :", output.shape, flush=True)

        loss = criterion(output, target)
        epoch_loss[0] += loss.detach()

        if args.adv and epoch >= args.adv_start:
            lag_out = output[:, :3]
            eul_out = lag2eul(
                lag_out,
                a=np.float64(style),
                eul_scale_factor=EUL_SCALE_FACTOR,
                inv_shuffle=True,
            )[0]
            lag_tgt = target[:, :3]
            eul_tgt = lag2eul(
                lag_tgt,
                a=np.float64(style),
                eul_scale_factor=EUL_SCALE_FACTOR,
                inv_shuffle=True,
            )[0]

            output = torch.cat([output, eul_out], dim=1)
            target = torch.cat([target, eul_tgt], dim=1)

            if args.cgan:
                output = torch.cat([input, output], dim=1)
                target = torch.cat([input, target], dim=1)
                # the output and target array is now [input, eul_out/eul_tgt, output/target]

            set_requires_grad(adv_model, True)

            score_out = adv_model(output.detach(), style=style)
            score_tgt = adv_model(target, style=style)
            adv_loss_real, adv_loss_fake = adv_criterion(score_out, score_tgt)

            epoch_loss[3] += adv_loss_fake.detach()
            epoch_loss[4] += adv_loss_real.detach()

            adv_optimizer.zero_grad(set_to_none=True)
            adv_loss_fake.backward()
            adv_loss_real.backward()

            adv_loss = adv_loss_fake + adv_loss_real
            epoch_loss[2] += adv_loss.detach()

            # if args.adv_wgan_gp_interval > 0 and batch % args.adv_wgan_gp_interval == 0:
            #     adv_loss_reg = wgan_grad_penalty(adv_model, output, target, style=style)
            #     adv_loss_reg_ = adv_loss_reg * args.adv_wgan_gp_interval
            #
            #     adv_loss_reg_.backward()
            #
            #     if batch % adv_wgan_gp_log_interval == 0 and rank == 0:
            #         logger.add_scalar(
            #             "train/batch/loss/adv/reg",
            #             adv_loss_reg.detach(),
            #             global_step=batch,
            #         )

            adv_optimizer.step()
            adv_grads = get_grads(adv_model)

            # generator adversarial loss
            if batch % args.adv_iter_ratio == 0:
                set_requires_grad(adv_model, False)

                score_out = adv_model(output, style=style)
                loss_adv = adv_criterion(score_out)
                epoch_loss[1] += args.adv_iter_ratio * loss_adv.detach()

                optimizer.zero_grad(set_to_none=True)
                loss_adv.backward()
                optimizer.step()
                grads = get_grads(model)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            grads = get_grads(model)

        if batch % args.log_interval == 0:
            dist.all_reduce(loss)
            loss /= world_size
            if rank == 0:
                logger.add_scalar(
                    "train/batch/loss/generator/l2", loss.detach(), global_step=batch
                )
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar(
                        "train/batch/loss/adv/G", loss_adv.detach(), global_step=batch
                    )
                    logger.add_scalars(
                        "train/batch/loss/adv/D",
                        {
                            "total": adv_loss.detach(),
                            "fake": adv_loss_fake.detach(),
                            "real": adv_loss_real.detach(),
                        },
                        global_step=batch,
                    )

                logger.add_scalar(
                    "train/batch/grad/generator/first", grads[0], global_step=batch
                )
                logger.add_scalar(
                    "train/batch/grad/generator/last", grads[-1], global_step=batch
                )
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar(
                        "train/batch/grad/adv/first", adv_grads[0], global_step=batch
                    )
                    logger.add_scalar(
                        "train/batch/grad/adv/last", adv_grads[-1], global_step=batch
                    )

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar(
            "train/epoch/loss/generator/l2", epoch_loss[0], global_step=epoch + 1
        )
        if args.adv and epoch >= args.adv_start:
            logger.add_scalar(
                "train/epoch/loss/adv/G", epoch_loss[1], global_step=epoch + 1
            )
            logger.add_scalars(
                "train/epoch/loss/adv/D",
                {
                    "total": epoch_loss[2],
                    "fake": epoch_loss[3],
                    "real": epoch_loss[4],
                },
                global_step=epoch + 1,
            )

        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)
            output = output[:, skip_chan:]
            target = target[:, skip_chan:]
        # input: 1, 6, 128, 128, 128
        # output: 1,
        try:
            with torch.no_grad():
                input_disp = input[-1, :3]
                input_vel = input[-1, 3:6]

                output_disp = output[-1, :3]
                output_vel = output[-1, 3:6]

                tgt_disp = target[-1, :3]
                tgt_vel = target[-1, 3:6]

                input_eul = lag2eul(
                    input_disp,
                    a=np.float64(style),
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
                output_eul = lag2eul(
                    output_disp,
                    a=np.float64(style),
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
                tgt_eul = lag2eul(
                    tgt_disp,
                    a=np.float64(style),
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
            fig = plt_slices(
                input_disp,
                output_disp,
                tgt_disp,
                output_disp - tgt_disp,
                input_vel,
                output_vel,
                tgt_vel,
                output_vel - tgt_vel,
                input_eul,
                output_eul,
                tgt_eul,
                output_eul - tgt_eul,
                title=[
                    "in disp",
                    "out disp",
                    "tgt disp",
                    "out disp - tgt disp",
                    "in vel",
                    "out vel",
                    "tgt vel",
                    "out vel - tgt vel",
                    "input eul",
                    "output eul",
                    "target eul",
                    "output eul - target eul",
                ],
                **args.misc_kwargs,
            )
            logger.add_figure("fig/train", fig, global_step=epoch + 1)
            fig.clf()
        except Exception as error:
            print(error)

        # fig = plt_power(
        #     input, output, target,
        #     label=['in', 'out', 'tgt'],
        #     **args.misc_kwargs,
        # )
        # logger.add_figure('fig/train/power/lag', fig, global_step=epoch+1)
        # fig.clf()
        # torch.cuda.memory_snapshot()

        # fig = plt_power(
        #     1.0,
        #     dis=[input, output, target],
        #     label=["in", "out", "tgt"],
        #     **args.misc_kwargs,
        # )
        # logger.add_figure("fig/train/power/eul", fig, global_step=epoch + 1)
        # fig.clf()

    return epoch_loss


def dist_init(rank, args):
    dist_file = "dist_addr"

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()

        args.dist_addr = "tcp://{}:{}".format(addr, port)

        with open(dist_file, mode="w") as f:
            f.write(args.dist_addr)
    else:
        while not os.path.exists(dist_file):
            time.sleep(1)

        with open(dist_file, mode="r") as f:
            args.dist_addr = f.read()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    dist.barrier()

    if rank == 0:
        os.remove(dist_file)


def init_weights(m):
    if isinstance(
        m,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        m.weight.data.normal_(0.0, args.init_weight_std)
    elif isinstance(
        m,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    ):
        if m.affine:
            m.weight.data.normal_(1.0, args.init_weight_std)
            m.bias.data.fill_(0)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """gradients of the weights of the first and the last layer"""
    grads = list(p.grad for n, p in model.named_parameters() if ".weight" in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm() for g in grads]
    return grads
