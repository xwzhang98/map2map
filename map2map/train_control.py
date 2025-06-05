import os
import socket
import sys
import time
from pprint import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
    hinge_grad_penalty,
    r1_regularization,
    ControlNet,
    MultiScaleControlNet,
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
    print("local_rank", local_rank)
    print("node", node)
    print("args.gpus_per_node", args.gpus_per_node)
    print("device_count", torch.cuda.device_count())
    print("args.world_size", args.world_size)
    print("gpu name", torch.cuda.get_device_name(local_rank))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)
    
    rank = args.gpus_per_node * node + local_rank
    torch.manual_seed(args.seed + rank)
    
    dist_init(rank, args, local_rank)

    # Create dataset with control inputs
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

    # Load pretrained generator
    base_generator = import_attr(args.base_model, models, callback_at=args.callback_at)
    base_generator = base_generator(
        sum(args.in_chan),
        sum(args.out_chan),
        style_size=args.style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )

    # Load pretrained generator weights if specified
    if hasattr(args, 'pretrained_generator_path') and args.pretrained_generator_path:
        if rank == 0:
            print(f"Loading pretrained generator from {args.pretrained_generator_path}")
        pretrained_state = torch.load(args.pretrained_generator_path, map_location=device)
        if 'model' in pretrained_state:
            load_model_state_dict(base_generator, pretrained_state['model'], strict=True)
        else:
            load_model_state_dict(base_generator, pretrained_state, strict=True)

    # Create ControlNet wrapper
    if hasattr(args, 'multi_scale_control') and args.multi_scale_control:
        # Multi-scale control with multiple control inputs
        control_channels_list = getattr(args, 'control_channels_list', [3, 3])  # Default: 2 control inputs of 3 channels each
        model = MultiScaleControlNet(
            generator=base_generator,
            control_channels_list=control_channels_list,
            control_scale_factors=getattr(args, 'control_scale_factors', None),
            use_normalize=getattr(args, 'control_normalize', False),
            freeze_generator=getattr(args, 'freeze_generator', True),
        )
    else:
        # Single control input
        control_in_chan = getattr(args, 'control_in_chan', 3)  # Default: 3 channels (e.g., density field)
        model = ControlNet(
            generator=base_generator,
            control_in_chan=control_in_chan,
            control_scale_factor=getattr(args, 'control_scale_factor', None),
            use_normalize=getattr(args, 'control_normalize', False),
            freeze_generator=getattr(args, 'freeze_generator', True),
        )

    model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], process_group=dist.new_group()
    )

    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    # Only train ControlNet parameters if generator is frozen
    if getattr(args, 'freeze_generator', True):
        # Get parameters excluding the frozen generator
        trainable_params = []
        for name, param in model.named_parameters():
            if 'generator' not in name or not param.requires_grad:
                if param.requires_grad:
                    trainable_params.append(param)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params_count = sum(p.numel() for p in trainable_params)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params_count:,}")
    else:
        trainable_params = model.parameters()

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        trainable_params,
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

        torch.set_rng_state(state["rng"].cpu())

        if rank == 0:
            min_loss = state["min_loss"]
            if args.adv and "adv_model" not in state:
                min_loss = None

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

        # Control strength scheduling
        if hasattr(args, 'control_strength_schedule') and args.control_strength_schedule:
            control_strength = min(1.0, epoch / getattr(args, 'control_warmup_epochs', 100))
            model.module.set_control_strength(control_strength)
            if rank == 0 and epoch % 10 == 0:
                print(f"Control strength: {control_strength:.3f}")

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
            os.symlink(state_file, tmp_link)
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()
    
    torch.cuda.empty_cache()


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
    MESHSIZE = args.target_meshsize
    epoch_start_time = time.time()

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

        # Extract control inputs from the data
        # Assume control inputs are included in the input or separately in data
        if hasattr(args, 'control_from_target') and args.control_from_target:
            # Use target as control (e.g., for style transfer)
            control = target
        elif hasattr(args, 'control_key') and args.control_key in data:
            # Control is provided separately in data
            control = data[args.control_key].to(device, non_blocking=True)
        else:
            # Use input as control (default behavior)
            control = input

        # Handle multi-scale control
        if hasattr(model.module, 'control_channels_list'):
            # Split control into multiple control inputs
            controls = []
            start_idx = 0
            for control_chan in model.module.control_channels_list:
                end_idx = start_idx + control_chan
                controls.append(control[:, start_idx:end_idx])
                start_idx = end_idx
            output = model(input, style, controls)
        else:
            # Single control input
            output = model(input, style, control)

        if i <= 5 or batch % 200 == 0 and rank == 0:
            print("##### batch :", batch)
            print("##### total batch :", len(loader))
            print("input shape :", input.shape)
            print("control shape :", control.shape if not isinstance(control, list) else [c.shape for c in control])
            print("output shape :", output.shape)
            print("target shape :", target.shape)
            print("style shape :", style.shape)

        if hasattr(model.module.generator, "scale_factor") and model.module.generator.scale_factor != 1:
            input_resampled = resample(input, model.module.generator.scale_factor, narrow=False)
            orig_input = input
            input = input_resampled
            del orig_input, input_resampled
            
        input_orig, output_orig, target_orig = input, output, target
        input, output, target = narrow_cast(input, output, target)
        if input.shape != input_orig.shape:
            del input_orig, output_orig, target_orig
        if i <= 5 and rank == 0:
            print("narrowed shape :", output.shape, flush=True)

        loss = criterion(output, target)
        epoch_loss[0] += loss.detach()

        if args.adv and epoch >= args.adv_start:
            current_scale = model.module.generator.scale_factor
            target_scale = 8
            
            if current_scale < target_scale:
                upscale_ratio = target_scale // current_scale
                print(f"Upsampling {current_scale}x to {target_scale}x (ratio: {upscale_ratio})")
                
                output_orig = output
                target_orig = target
                input_orig = input
                
                output = F.interpolate(
                    output_orig, 
                    scale_factor=upscale_ratio, 
                    mode='trilinear', 
                    align_corners=False
                )
                target = F.interpolate(
                    target_orig,
                    scale_factor=upscale_ratio,
                    mode='trilinear', 
                    align_corners=False
                )
                input = F.interpolate(
                    input_orig,
                    scale_factor=upscale_ratio,
                    mode='trilinear', 
                    align_corners=False
                )
                
                del output_orig, target_orig, input_orig
            
            with torch.set_grad_enabled(True):
                lag_out = output[:, :3]
                lag_tgt = target[:, :3]
                
                eul_out = lag2eul(
                    lag_out,
                    a=np.float64(style),
                    meshsize=MESHSIZE,
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=True,
                )[0]
                
                eul_tgt = lag2eul(
                    lag_tgt,
                    a=np.float64(style),
                    meshsize=MESHSIZE,
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=True,
                )[0]
                
                output_orig = output
                target_orig = target
                
                output = torch.cat([output, eul_out], dim=1)
                target = torch.cat([target, eul_tgt], dim=1)
                
                del lag_out, lag_tgt, eul_out, eul_tgt
                
                if output_orig.shape != output.shape:
                    del output_orig, target_orig

                if args.cgan:
                    output_orig = output
                    target_orig = target
                    
                    output = torch.cat([input, output], dim=1)
                    target = torch.cat([input, target], dim=1)
                    
                    if output_orig.shape != output.shape:
                        del output_orig, target_orig

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

            if args.adv_wgan_gp_interval > 0 and batch % args.adv_wgan_gp_interval == 0:
                adv_loss_reg = r1_regularization(adv_model, target, style=style)
                adv_loss_reg_ = adv_loss_reg * args.adv_wgan_gp_interval

                adv_loss_reg_.backward()

                if batch % adv_wgan_gp_log_interval == 0 and rank == 0:
                    logger.add_scalar(
                        "Loss/Discriminator/R1_Penalty",
                        adv_loss_reg.detach(),
                        global_step=batch,
                    )
                    logger.add_scalar(
                        "Loss/Discriminator/R1_Scaled",
                        adv_loss_reg_.detach(),
                        global_step=batch,
                    )

            adv_optimizer.step()
            adv_grads = get_grads(adv_model)

            if batch % args.adv_iter_ratio == 0:
                set_requires_grad(adv_model, False)

                score_out = adv_model(output, style=style)
                loss_adv = adv_criterion(score_out)
                epoch_loss[1] += args.adv_iter_ratio * loss_adv.detach()

                optimizer.zero_grad(set_to_none=True)
                (0.01*loss + loss_adv).backward()
                optimizer.step()
                try:
                    grads = get_grads(model)
                except Exception as e:
                    grads = [0, 0]
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            try:
                grads = get_grads(model)
            except Exception as e:
                grads = [0, 0]

        if batch % args.log_interval == 0:
            dist.all_reduce(loss)
            loss /= world_size
            if rank == 0:
                logger.add_scalar(
                    "Loss/Generator/L2", loss.detach(), global_step=batch
                )
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar(
                        "Loss/Generator/Adversarial", loss_adv.detach(), global_step=batch
                    )
                    logger.add_scalar(
                        "Loss/Generator/Total", (0.01*loss + loss_adv).detach(), global_step=batch
                    )
                    logger.add_scalar(
                        "Loss/Discriminator/Total", adv_loss.detach(), global_step=batch
                    )
                    logger.add_scalar(
                        "Loss/Discriminator/Fake", adv_loss_fake.detach(), global_step=batch
                    )
                    logger.add_scalar(
                        "Loss/Discriminator/Real", adv_loss_real.detach(), global_step=batch
                    )
                    with torch.no_grad():
                        d_real_acc = (score_tgt > 0).float().mean()
                        d_fake_acc = (score_out < 0).float().mean()
                        d_total_acc = (d_real_acc + d_fake_acc) / 2
                    logger.add_scalar(
                        "Metrics/Discriminator/RealAccuracy", d_real_acc, global_step=batch
                    )
                    logger.add_scalar(
                        "Metrics/Discriminator/FakeAccuracy", d_fake_acc, global_step=batch
                    )
                    logger.add_scalar(
                        "Metrics/Discriminator/TotalAccuracy", d_total_acc, global_step=batch
                    )

                # Control-specific metrics
                if hasattr(args, 'control_strength_schedule') and args.control_strength_schedule:
                    control_strength = min(1.0, epoch / getattr(args, 'control_warmup_epochs', 100))
                    logger.add_scalar(
                        "Control/Strength", control_strength, global_step=batch
                    )

                logger.add_scalar(
                    "Gradients/Generator/FirstLayer", grads[0], global_step=batch
                )
                logger.add_scalar(
                    "Gradients/Generator/LastLayer", grads[-1], global_step=batch
                )
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar(
                        "Gradients/Discriminator/FirstLayer", adv_grads[0], global_step=batch
                    )
                    logger.add_scalar(
                        "Gradients/Discriminator/LastLayer", adv_grads[-1], global_step=batch
                    )
                    grad_ratio = grads[-1] / (adv_grads[-1] + 1e-8)
                    logger.add_scalar(
                        "Gradients/G_D_Ratio", grad_ratio, global_step=batch
                    )
                
                logger.add_scalar(
                    "LearningRate/Generator", optimizer.param_groups[0]['lr'], global_step=batch
                )
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar(
                        "LearningRate/Discriminator", adv_optimizer.param_groups[0]['lr'], global_step=batch
                    )

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar(
            "Epoch/Loss/Generator/L2", epoch_loss[0], global_step=epoch + 1
        )
        if args.adv and epoch >= args.adv_start:
            logger.add_scalar(
                "Epoch/Loss/Generator/Adversarial", epoch_loss[1], global_step=epoch + 1
            )
            logger.add_scalar(
                "Epoch/Loss/Discriminator/Total", epoch_loss[2], global_step=epoch + 1
            )
            logger.add_scalar(
                "Epoch/Loss/Discriminator/Fake", epoch_loss[3], global_step=epoch + 1
            )
            logger.add_scalar(
                "Epoch/Loss/Discriminator/Real", epoch_loss[4], global_step=epoch + 1
            )
            loss_balance = epoch_loss[1] / (epoch_loss[2] + 1e-8)
            logger.add_scalar(
                "Epoch/Metrics/LossBalance_G_D", loss_balance, global_step=epoch + 1
            )

        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)
            output = output[:, skip_chan:]
            target = target[:, skip_chan:]

        try:
            with torch.no_grad():
                input_disp = input[-1, :3][None, :]
                input_vel = input[-1, 3:6][None, :]

                output_disp = output[-1, :3][None, :]
                output_vel = output[-1, 3:6][None, :]

                tgt_disp = target[-1, :3][None, :]
                tgt_vel = target[-1, 3:6][None, :]

                input_eul = lag2eul(
                    input_disp,
                    a=np.float64(style),
                    meshsize=MESHSIZE,
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
                output_eul = lag2eul(
                    output_disp,
                    a=np.float64(style),
                    meshsize=MESHSIZE,
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
                tgt_eul = lag2eul(
                    tgt_disp,
                    a=np.float64(style),
                    meshsize=MESHSIZE,
                    eul_scale_factor=EUL_SCALE_FACTOR,
                    inv_shuffle=False,
                )[0]
            fig = plt_slices(
                input_disp[-1],
                output_disp[-1],
                tgt_disp[-1],
                output_disp[-1] - tgt_disp[-1],
                input_vel[-1],
                output_vel[-1],
                tgt_vel[-1],
                output_vel[-1] - tgt_vel[-1],
                input_eul[-1],
                output_eul[-1],
                tgt_eul[-1],
                output_eul[-1] - tgt_eul[-1],
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
            logger.add_figure("Visualization/Fields", fig, global_step=epoch + 1)
            fig.clf()
            
            if epoch % 5 == 0:
                with torch.no_grad():
                    disp_rel_error = torch.norm(output_disp - tgt_disp) / (torch.norm(tgt_disp) + 1e-8)
                    vel_rel_error = torch.norm(output_vel - tgt_vel) / (torch.norm(tgt_vel) + 1e-8)
                    
                    logger.add_scalar(
                        "Physics/RelativeError/Displacement", disp_rel_error, global_step=epoch + 1
                    )
                    logger.add_scalar(
                        "Physics/RelativeError/Velocity", vel_rel_error, global_step=epoch + 1
                    )
                    
                    logger.add_scalar(
                        "Physics/OutputStd/Velocity", output_vel.std(), global_step=epoch + 1
                    )
                
        except Exception as error:
            print("Error encountered in plotting/metrics: ", error)

        if epoch % 50 == 0:
            for name, param in model.named_parameters():
                if ('conv' in name or 'fc' in name) and 'weight' in name and param.grad is not None:
                    layer_parts = name.split('.')
                    if any(x in layer_parts[0] for x in ['0', '1', '2']) or any(x in layer_parts[-2] for x in ['final', 'out', 'last']):
                        logger.add_histogram(f'Weights/ControlNet/{name}', param.data, global_step=epoch + 1)
            
            if args.adv and epoch >= args.adv_start:
                for name, param in adv_model.named_parameters():
                    if ('conv' in name or 'fc' in name) and 'weight' in name and param.grad is not None:
                        layer_parts = name.split('.')
                        if any(x in layer_parts[0] for x in ['0', '1', '2']) or any(x in layer_parts[-2] for x in ['final', 'out', 'last']):
                            logger.add_histogram(f'Weights/Discriminator/{name}', param.data, global_step=epoch + 1)
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            logger.add_scalar("System/GPU_Memory_Allocated_GB", memory_allocated, global_step=epoch + 1)
            logger.add_scalar("System/GPU_Memory_Cached_GB", memory_cached, global_step=epoch + 1)
        
        logger.add_scalar("System/Epoch_Duration_Minutes", (time.time() - epoch_start_time) / 60, global_step=epoch + 1)

    return epoch_loss


def dist_init(rank, args, local_rank):
    dist_file = "dist_addr"
    
    print(f"Rank {rank}: Setting up distributed training with local_rank {local_rank}")
    
    if rank == 0:
        addr = socket.gethostname()
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()
        
        args.dist_addr = "tcp://{}:{}".format(addr, port)
        print(f"Rank {rank}: Master node using address {args.dist_addr}")
        
        with open(dist_file, mode="w") as f:
            f.write(args.dist_addr)
    else:
        timeout_count = 0
        while not os.path.exists(dist_file):
            time.sleep(1)
            timeout_count += 1
            if timeout_count > 60:
                raise TimeoutError(f"Rank {rank}: Timed out waiting for dist_file")
        
        with open(dist_file, mode="r") as f:
            args.dist_addr = f.read()
        print(f"Rank {rank}: Using address {args.dist_addr} from master")
    
    print(f"Rank {rank}: Initializing process group")
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    
    print(f"Rank {rank}: Waiting at barrier with device_ids=[{local_rank}]")
    dist.barrier(device_ids=[local_rank])
    print(f"Rank {rank}: Passed barrier")
    
    if rank == 0:
        os.remove(dist_file)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    grads = list(p.grad for n, p in model.named_parameters() if ".weight" in n and p.grad is not None)
    if len(grads) >= 2:
        grads = [grads[0], grads[-1]]
    elif len(grads) == 1:
        grads = [grads[0], grads[0]]
    else:
        grads = [torch.tensor(0.0), torch.tensor(0.0)]
    grads = [g.detach().norm() for g in grads]
    return grads