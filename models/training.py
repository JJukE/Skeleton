import torch
from jjuke.network_utils.distributed import reduce_dict


class BaseTrainer(object):
    """ Base trainer for all networks. """
    def __init__(self, cfg, net, optimizer, device=None):
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.device = device

    def set_mode(self, phase):
        for net_type, subnet in self.net.items():
            # set mode
            subnet.train(phase == "train")
            # set subnet mode for frozen layers
            if self.cfg.config.distributed.use_ddp:
                subnet.module.set_mode()
            else:
                subnet.set_mode()

    def show_net_n_params(self):
        for net_type, subnet in self.net.items():
            self.cfg.info("Total number of parameters in {0:s}: {1:d}.".format(net_type, sum(
                p.numel() for p in subnet.parameters())))
            if self.cfg.config.distributed.use_ddp:
                for name, module in subnet.module.named_children():
                    self.cfg.info("--- module name - {0:s}: {1:d}.".format(name, sum(p.numel() for p in module.parameters())))
            else:
                for name, module in subnet.named_children():
                    self.cfg.info("--- module name - {0:s}: {1:d}.".format(name, sum(p.numel() for p in module.parameters())))

    def show_lr(self):
        """ Display current learning rates """
        for net_type in self.optimizer:
            lrs = [self.optimizer[net_type].param_groups[i]["lr"] for i in range(len(self.optimizer[net_type].param_groups))]
            self.cfg.info("Current {} learning rates are: ".format((net_type) + str(lrs) + "."))

    def clip_grad_norm(self, net):
        if self.cfg.config.distributed.use_ddp:
            for module in net.module.children():
                if hasattr(module, "optim_spec") and module.optim_spec["clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), module.optim_spec["clip_norm"])
        else:
            for module in net.children():
                if hasattr(module, "optim_spec") and module.optim_spec["clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), module.optim_spec["clip_norm"])

    def train_step(self, data, stage="all"):
        """ Performs a step training

        Args:
            data (dict): data dictionary
        """
        self.optimizer["generator"].zero_grad()
        loss = self.compute_loss(data)
        if loss["total"].requires_grad:
            loss["total"].backward()
            if self.cfg.config.train.clip_norm:
                self.clip_grad_norm(net=self.net["generator"])
            self.optimizer["generator"].step()

        # for logging
        loss_reduced = reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict

    def eval_loss_parser(self, loss_recorder):
        """ Get the eval
        
        Args:
            loss_recorder: loss recorder for all losses.
        """
        return loss_recorder["total"].avg

    def compute_loss(self, *args, **kwargs):
        """ Performs a training step. """
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs an evaluation step. """
        raise NotImplementedError

    def visualize_step(self, *args, **kwargs):
        """ Performs a visualization step. """
        if not self.cfg.is_master:
            return
        raise NotImplementedError