import sys

import torch
from torch import nn

from models.registers import METHODS
from models import method_paths


def load_device(cfg):
    '''
    load device settings
    '''
    if cfg.config.device.use_gpu and torch.cuda.is_available():
        cfg.info('GPU mode is on.')
        return torch.device(torch.cuda.current_device())
    else:
        cfg.info('CPU mode is on.')
        return torch.device("cpu")

def load_model(cfg, device):
    '''
    load specific network from configuration file
    :param cfg: configuration file
    :param device: torch.device
    :return:
    '''
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
    '''
    load trainer for training and validation
    :param cfg: configuration file
    :param net: nn.Module network
    :param optimizer: torch.optim
    :param device: torch.device
    :return:
    '''
    trainer = method_paths[cfg.config.method].config.get_trainer(cfg=cfg,
                                                                 net=net,
                                                                 optimizer=optimizer,
                                                                 device=device)
    return trainer

def load_evaluater(cfg, net, device):
    '''
    load evaluater for evaluation
    :param cfg: configuration file
    :param net: nn.Module network
    :param device: torch.device
    :return:
    '''
    evaluater = method_paths[cfg.config.method].config.get_evaluater(
        cfg=cfg,
        net=net,
        device=device)
    return evaluater

def load_dataloader(cfg, mode):
    '''
    load dataloader
    :param cfg: configuration file.
    :param mode: 'train', 'val' or 'eval'.
    :return:
    '''
    dataloader = method_paths[cfg.config.method].config.get_dataloader(cfg=cfg, mode=mode)
    return dataloader
