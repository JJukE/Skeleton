# This is for example.
from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork


@METHODS.register_module
class VAE(BaseNetwork):
    def __init__(self, cfg, net_type, device):
        """ Load submodules for the network.
        
        Args:
            config: customized configurations.
        """
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.device = device
        config = cfg.config

        phase_names = []
        if config[config.mode].phase in ["full", "generation"]:
            phase_names += ["latent_encode"]

        if (not config.model[net_type].arch) or (not phase_names):
            cfg.info("No submodule found. Please check the phase name and model definition.")
            raise ModuleNotFoundError("No submodule found. Please check the phase name and model definition.")

        """ Load network blocks """
        for phase_name, net_spec in config.model[net_type].arch.items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec["module"]
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(config, net_spec, net_type)
            subnet = MODULES.get(method_name)(cfg, optim_spec, device)
            self.add_module(phase_name, subnet)

            """ Load corresponding loss functions """
            setattr(self, phase_name + "_loss", LOSSES.get(config.model[net_type].arch[phase_name].loss, "Null")(
                config.model[net_type].arch[phase_name].get("weight", 1), cfg, device))

        """ Freeze submodules or not """
        self.freeze_modules(config)

    def interpolate(self, idx1, idx2, interval):
        interpolated = self.latent_encode.interpolate(idx1, idx2, interval)
        return interpolated

    def sample_latent(self):
        interpolated = self.latent_encode.sample_latent()
        return interpolated

    def forward(self, data):
        latent_z = self.latent_encode(data["room_idx"][:, 0])
        return latent_z

    def loss(self, *args, **kwargs):
        raise NotImplementedError


@METHODS.register_module
class Generator(BaseNetwork):
    def __init__(self, cfg, net_type, device):
        """ Load submodules for the network.
        
        Args:
            config: customized configurations.
        """
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.device = device
        config = cfg.config

        phase_names = []
        if config[config.mode].phase in ["full"]:
            phase_names += ["backbone", "box_gen", "render"]
        else:
            phase_names += ["backbone", "box_gen"]

        if config.start_deform:
            phase_names.append("shape_gen")

        if (not config.model[net_type].arch) or (not phase_names):
            cfg.info("No submodule found. Please check the phase name and model definition.")
            raise ModuleNotFoundError("No submodule found. Please check the phase name and model definition.")

        """ Load network blocks """
        for phase_name, net_spec in config.model[net_type].arch.items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec["module"]
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(config, net_spec, net_type)
            subnet = MODULES.get(method_name)(cfg, optim_spec, device)
            self.add_module(phase_name, subnet)

            """ Load corresponding loss functions """
            setattr(self, phase_name + "_loss", LOSSES.get(config.model[net_type].arch[phase_name].loss, "Null")(
                config.model[net_type].arch[phase_name].get("weight", 1), cfg, device))

        """ Freeze submodules or not """
        self.freeze_modules(config)

    def generate(self, latent_z, data, start_deform=False, output_render=True, **kwargs):
        """ Sample boxes on images.

        Args:
            max_n_preds: maximal number of predictions.
        """
        backbone_feat = self.backbone.generate_boxes(latent_z, data["room_type_idx"], **kwargs)
        box3ds = self.box_gen.generate(backbone_feat)
        if start_deform:
            meshes = self.shape_gen.generate(backbone_feat)
            render_mask_tr = data.get("render_mask_tr", None)
        else:
            meshes = []
            for feat in backbone_feat:
                n_batch, n_object = feat.shape[:2]
            render_mask_tr = None

        if output_render:
            outputs = self.render.generate(box3ds, meshes, data["cam_T"], data["cam_K"], data["image_size"], render_mask_tr,
                                           start_deform=start_deform, **kwargs)
        else:
            outputs = []
            for boxes_per_scene, meshes_per_scene in zip(box3ds, meshes):
                centers = boxes_per_scene[..., :3]
                sizes = boxes_per_scene[..., 3:6]
                outputs.append(
                    {"box3ds": boxes_per_scene})

        return outputs

    def forward(self, latent_z, data, start_deform=False, **kwargs):
        """ Forward pass of the network
        
        Args:
            data (dict): contains the data for training.
        Returns:
            end_points: dict
        """
        if (self.cfg.config.mode == "demo" and self.cfg.config.data.n_views == 1) or (
                self.cfg.config.mode == "test" and self.cfg.config.test.n_views_for_finetune == 1):
            max_len = self.cfg.max_n_obj
        else:
            max_len = data["max_len"].max()
        backbone_feat, completeness_score = self.backbone(latent_z, max_len, data["room_type_idx"])

        box3ds = self.box_gen(backbone_feat, completeness_score)

        if start_deform:
            meshes = self.shape_gen(backbone_feat)
            render_mask_tr = data.get("render_mask_tr", None)
        else:
            n_batch, n_object = backbone_feat.shape[:2]
            render_mask_tr = None

        # generate cameras
        renderings = self.render(box3ds, meshes, data["cam_T"], data["cam_K"], data["image_size"],
                                 render_mask_tr, start_deform=start_deform, pred_mask=data["max_len"][:, 0], **kwargs)

        return renderings

    def loss(self, pred_data, gt_data, start_deform=False, **kwargs):
        """ Calculate loss of est_out given gt_out. """
        render_loss, extra_output = self.render_loss(pred_data, gt_data, start_deform=start_deform, **kwargs)
        return render_loss, extra_output