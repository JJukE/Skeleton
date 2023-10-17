import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, cfg, mode):
        """ Initiate a base dataset for data loading in other networks.
        This is just an example, it can be changed for common dataset code for various [Method]s
        or even nothing if all methods has each different dataset codes. In this case, dataset
        code will be in the "./Method/dataloader.py".

        Args:
            cfg: config file
            mode: train/val/eval mode
        """
        self.cfg = cfg
        self.mode = mode
        self._room_uids = cfg.room_uids[self.mode]
        self._split = cfg.split_data[mode]

    @property
    def split(self):
        return self._split

    @property
    def room_uids(self):
        return self._room_uids

    def update_split(self, n_views_for_finetune=-1, room_uid=None, **kwargs):
        if n_views_for_finetune > 0:
            self.sample_n_views_per_scene(n_views_for_finetune)

        if room_uid is not None:
            if "aug" not in room_uid:
                room_uid = room_uid+"_aug_0"
            self._split = {room_uid: self.split[room_uid]}
            self._room_uids = [room_uid]

    def sample_n_views_per_scene(self, n_views_for_finetune):
        """Sample n views per scene."""
        for room_uid in self.room_uids:
            samples_in_room = self.split[room_uid]
            if len(samples_in_room) > n_views_for_finetune:
                samples_in_room = np.random.choice(samples_in_room, n_views_for_finetune, replace=False).tolist()
            self._split[room_uid] = samples_in_room

    def __len__(self):
        return len(self.room_uids)