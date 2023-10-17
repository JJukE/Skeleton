# This is for example
import torch
from torch import nn

from models.registers import MODULES


@MODULES.register_module
class ARTransformer(nn.Module):
    def __init__(self, cfg, optim_spec=None, device="cuda"):
        """
        Encode scene priors from embeddings

        Args:
            cfg: configuration file.
            optim_spec: optimizer parameters.
        """
        super().__init__()
        """ Optimizer parameters used in training """
        self.optim_spec = optim_spec
        self.device = device

        """ Network """
        # Parameters
        self.z_dim = cfg.config.data.z_dim
        self.inst_latent_len = cfg.config.data.backbone_latent_len
        self.max_obj_num = cfg.max_n_obj
        d_model = 512
        n_head = 4

        # Build Networks
        # empty room token in transformer encoder
        self.empty_token_embedding = nn.Embedding(len(cfg.room_types), self.z_dim)

        # Build a transformer encoder
        self.transformer_encoder # = ...

        self.transformer_decoder # = ...

        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU()) for _ in range(self.max_obj_num)])

        self.mlp_bbox = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.inst_latent_len))
        self.mlp_comp = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, latent_z, max_len, room_type_idx):
        obj_feats = [self.empty_token_embedding(room_type_idx[:, 0])[:, None]]

        for idx in range(self.max_obj_num):
            X = torch.cat(obj_feats, dim=1)
            # if idx > 0:
            #     X = X.detach()
            X = self.transformer_encoder(X, length_mask=None)
            last_feat = self.transformer_decoder(latent_z, X)
            obj_feats.append(self.encoders[idx](last_feat))

        obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :max_len]
        box_feat = self.mlp_bbox(obj_feats)
        completenesss_feat = self.mlp_comp(obj_feats)
        return box_feat, completenesss_feat

    @torch.no_grad()
    def generate_boxes(self, latent_codes, room_type_idx, pred_gt_matching=None, self_end=False, threshold=0.5, **kwargs):
        """ Generate boxes from latent codes """
        if pred_gt_matching is None:
            self_end = True

        assert self_end != (pred_gt_matching is not None)

        n_batch = latent_codes.size(0)
        output_feats = []
        for batch_id in range(n_batch):
            latent_z = latent_codes[[batch_id]]

            # prepare encoder input: start with empty token, will accumulate with more objects
            obj_feats = [self.empty_token_embedding(room_type_idx[[batch_id], 0])[:, None]]

            for idx in range(self.max_obj_num):
                X = torch.cat(obj_feats, dim=1)
                X = self.transformer_encoder(X, length_mask=None)
                last_feat = self.transformer_decoder(latent_z, X)
                last_feat = self.encoders[idx](last_feat)
                obj_feats.append(last_feat)

                if self_end:
                    completeness = self.mlp_comp(last_feat).sigmoid()
                    if completeness > threshold:
                        break

            obj_feats = torch.cat(obj_feats[1:], dim=1)
            box_feat = self.mlp_bbox(obj_feats)

            if pred_gt_matching is not None:
                box_feat = box_feat[:, pred_gt_matching[batch_id][0]]

            output_feats.append(box_feat)

        return output_feats