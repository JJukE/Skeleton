from .dataloader import my_dataloader
from .evaluating import Evaluater
from .training import Trainer

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)


def get_evaluater(cfg, net, device=None):
    return Evaluater(cfg=cfg, net=net, device=device)


def get_dataloader(cfg, mode):
    return my_dataloader(cfg=cfg, mode=mode)
