from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from models.datasets import BaseDataset # base or commonly used dataset class


class CustomDataset(BaseDataset):
    """ Example dataset class """
    def __init__(self, cfg, mode):
        super().__init__()
    
    def __getitem__(self, idx):
        pass


def my_worker_init_fn(cfg, worker_id):
    if cfg.config.use_random_seed:
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    else:
        np.random.seed(cfg.config.seed + worker_id)

def my_dataloader(cfg, mode):
    dataset = CustomDataset(cfg=cfg, mode=mode) # example

    if cfg.config.distributed.num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
    else:
        if mode == "train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    
    batch_sampler = BatchSampler(sampler,
                                 batch_size=cfg.config[mode].batch_size // cfg.config.distributed.num_gpus,
                                 drop_last=True)
    
    partial_worker_init_fn = partial(my_worker_init_fn, cfg=cfg)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=cfg.config.device.num_workers,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=partial_worker_init_fn)
    
    print("Loaded {} data for {} with {} property.".format(
        len(dataset), mode, dataset.ex_property
    ))

    return dataloader