import torch

from models.training import BaseTrainer
from jjuke.network_utils.distributed import reduce_dict


class Trainer(BaseTrainer):
    """ Trainer object """
    def __init__(self, cfg, net, optimizer, device=None):
        super().__init__(cfg, net, optimizer, device)

        self.latent_input = net["latent_input"] # 
        self.generator = net["generator"]


    @torch.no_grad()
    def eval_step(self, data):
        """ Performs a step in evaluation

        Args:
            data (dict): data dictionary
        """
        """ Load input and ground-truth data """
        # data = self.to_device(data)

        """ Network forwarding """
        latent_z = self.latent_input(data)
        est_data = self.generator(latent_z, data)

        """ Compute losses """
        if self.cfg.config.distributed.use_ddp:
            loss = self.generator.module.loss(est_data, data)
        else:
            loss = self.generator.loss(est_data, data)

        # for logging
        loss_reduced = reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict


    def train_step(self, data, stage="all", start_deform=False, **kwargs):
        """ Performs a step training

        Args:
            data (dict): data dictionary
        """
        if stage == "all":
            net_types = self.optimizer.keys()
        elif stage == "latent_only":
            net_types = ["latent_input"]
        else:
            raise ValueError("No such stage.")

        for net_type in net_types:
            self.optimizer[net_type].zero_grad()

        loss, extra_output = self.compute_loss(data, start_deform=start_deform, **kwargs)

        if loss["total"].requires_grad:
            loss["total"].backward()
            if self.cfg.config.train.clip_norm:
                self.clip_grad_norm(net=self.net["generator"])
            for net_type in net_types:
                self.optimizer[net_type].step()

        # for logging
        loss_reduced = reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict, extra_output


    @torch.no_grad()
    def visualize_step(self, *args, **kwargs):
        """ Performs a visualization step. """
        if not self.cfg.is_master:
            return
        pass


    def compute_loss(self, data, start_deform=False, **kwargs):
        """ Compute the overall loss.

        Args:
            data (dict): data dictionary
        """
        """ Load input and ground-truth data """
        # data = self.to_device(data)

        """ Network forwarding """
        latent_z = self.latent_input(data)
        est_data = self.generator(latent_z, data, start_deform=start_deform, **kwargs)

        """ Compute losses """
        if self.cfg.config.distributed.use_ddp:
            loss, extra_output = self.generator.module.loss(est_data, data, start_deform=start_deform, **kwargs)
        else:
            loss, extra_output = self.generator.loss(est_data, data, start_deform=start_deform, **kwargs)
        return loss, extra_output
