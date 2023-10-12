from scipy import stats
import torch
from torch.nn import L1Loss, CrossEntropyLoss, BCEWithLogitsLoss, CosineSimilarity, BCELoss
import torch.distributions as dist
from models.registers import LOSSES
from torch.nn import functional as F


class BaseLoss(object):
    ''' Base loss class. '''
    def __init__(self, weight=1, cfg=None, device='cuda'):
        ''' Initialize loss module '''
        super().__init__()
        self.weight = weight
        self.cfg = cfg
        self.device = device


@LOSSES.register_module
class Null(BaseLoss):
    ''' This loss function is for modules where a loss preliminary calculated. '''
    def __call__(self, loss):
        return self.weight * torch.mean(loss)


# @LOSSES.register_module
# class KL(BaseLoss):
#     def __init__(self, weight=1, cfg=None, device='cuda'):
#         super().__init__(weight=weight, cfg=cfg, device=device)
#         self.z_dim = cfg.config.data.z_dim
#         self.device = device

#     def get_prior_z(self, z_dim, device):
#         ''' Returns prior distribution for latent code z.
#         Args:
#             zdim: dimension of latent code z.
#             device (device): pytorch device
#         '''
#         p0_z = dist.Normal(
#             torch.zeros(z_dim, device=device),
#             torch.ones(z_dim, device=device)
#         )

#         return p0_z

#     def __call__(self, latent_dist):
#         p0_z = self.get_prior_z(self.z_dim, self.device)
#         kl = dist.kl_divergence(latent_dist, p0_z).sum(dim=-1)
#         kl = kl.mean()
#         return {'total': kl * self.weight, 'kl': kl}


# @LOSSES.register_module
# class 