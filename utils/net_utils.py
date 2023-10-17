import sys

import torch
from torch import nn

from models.registers import METHODS
from models import method_paths


def load_device(cfg):
    """
    Load device settings
    """
    if cfg.config.device.use_gpu and torch.cuda.is_available():
        cfg.info('GPU mode is on.')
        return torch.device(torch.cuda.current_device())
    else:
        cfg.info('CPU mode is on.')
        return torch.device("cpu")

def load_model(cfg, device):
    """ Load specific network from configuration file

    Args:
        cfg: configuration file
        device: torch.device
    """
    net = {}
    for net_type, net_specs in cfg.config.model.items():
        if net_specs.method not in METHODS.module_dict:
            cfg.info('The method %s is not defined, please check the correct name.' % (net_specs.method))
            cfg.info('Exit now.')
            sys.exit(0)

        model = METHODS.get(net_specs.method)(cfg, net_type, device)
        model.to(device)

        if cfg.config.distributed.num_gpus > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device,
                                                        static_graph=True)
        else:
            model = nn.DataParallel(model)

        net[net_type] = model

    return net

def load_trainer(cfg, net, optimizer, device):
    """ Load trainer for training and validation

    Args:
        cfg: configuration file
        net: nn.Module network
        optimizer: torch.optim
        device: torch.device
    """
    trainer = method_paths[cfg.config.method].config.get_trainer(cfg=cfg,
                                                                 net=net,
                                                                 optimizer=optimizer,
                                                                 device=device)
    return trainer

def load_evaluater(cfg, net, device):
    """ Load evaluater for evaluation

    Args:
        cfg: configuration file
        net: nn.Module network
        device: torch.device
    """
    evaluater = method_paths[cfg.config.method].config.get_evaluater(
        cfg=cfg,
        net=net,
        device=device)
    return evaluater

def load_dataloader(cfg, mode):
    """ Load dataloader

    Args:
        cfg: configuration file.
        mode: 'train', 'val' or 'eval'.
    """
    dataloader = method_paths[cfg.config.method].config.get_dataloader(cfg=cfg, mode=mode)
    return dataloader
