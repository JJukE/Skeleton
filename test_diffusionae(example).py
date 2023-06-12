import os
import random

import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.example_dataset import ShapeNetDataset
from models.example_model import DiffusionAE

from options.example_options import DiffusionAETestOptions

from jjuke.utils import seed_everything, print_model
from jjuke.utils.vis3d import ObjectVisualizer
from jjuke.logger import CustomLogger
from jjuke.metrics import EMD_CD

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Evaluation
#============================================================

@torch.no_grad()
def evaluate(model, test_loader, args, ckpt_args, res_dir):
    model.eval()
    
    all_ref = []
    all_recons = []
    all_labels = []
    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        label = batch['cate']
        
        code = model.encode(ref) # (B, z_dim)
        recons = model.decode(code, flexibility=ckpt_args.flexibility) # (B, num_points, 3)
        
        ref = ref * scale + shift
        recons = recons * scale + shift
        
        all_ref.append(ref.detach().cpu())
        all_recons.append(recons.detach().cpu())
        all_labels.append(label)
    
    all_ref = torch.cat(all_ref, dim=0) # (num_all_objects, num_points, 3)
    all_recons = torch.cat(all_recons, dim=0) # (num_all_objects, num_points, 3)
    all_labels = np.concatenate(all_labels, axis=0) # (num_all_objects)
    
    if args.visualize:
        visualizer = ObjectVisualizer()
        
        logger.info("Saving point clouds...")
        ref_to_save = all_ref[:args.num_vis]
        recon_to_save = all_recons[:args.num_vis]
        label_to_save = all_labels[:args.num_vis]
        
        visualizer.save(os.path.join(res_dir, "references.ply"), ref_to_save)
        visualizer.save(os.path.join(res_dir, "recons.ply"), recon_to_save)
        np.save(os.path.join(res_dir, "labels.npy"), label_to_save)
    
    logger.info("Computing metrics...")
    metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.test_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    # if args.visualize: # visualize on the window
    #     visualizer.visualize(all_ref[:args.num_vis], num_in_row=8)
    logger.info("[Eval] CD {:.12f} | EMD {:.12f}".format(cd, emd))

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for training
    args, arg_msg, device_msg = DiffusionAETestOptions().parse()

    if args.debug:
        args.data_dir = '/root/hdd1/G2S'
        args.data_name = '3rlabel_shapenetdata.hdf5'
        args.name = 'G2S_DPMPC_practice_230527'
        args.gpu_ids = '0' # only 0 is available while debugging
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.ckpt_name = 'ckpt_100000.pt'
        args.test_batch_size = 128
        args.use_randomseed = False
        args.visualize = True

    # get logger
    exp_dir = os.path.join(args.exps_dir, args.name, "ckpts")
    ckpt_path = os.path.join(exp_dir, args.ckpt_name)
    res_dir = os.path.join(args.exps_dir, args.name, "results")
    
    logger = CustomLogger(res_dir, isTrain=args.isTrain)
    
    logger.info(arg_msg)
    logger.info(device_msg)
    
    
    # set seed
    if not args.use_randomseed:
        args.seed = random.randint(1, 10000)
    logger.info("Seed: {}".format(args.seed))
    seed_everything(args.seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    dataset_path = os.path.join(args.data_dir, args.data_name)
    test_dataset = ShapeNetDataset(
        path=dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.scale_mode,
    )

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.test_batch_size,
                            num_workers=0)

    # Model
    logger.info("Loading model...")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = DiffusionAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    
    logger.info(print_model(model, verbose=args.verbose))

    # Main loop
    logger.info("Start evaluation...")
    try:
        evaluate(model, test_loader, args, ckpt_args=ckpt['args'], res_dir=res_dir)
        logger.info("Evaluation completed!")
        logger.flush()

    except KeyboardInterrupt:
        logger.info("Terminating...")
        logger.flush()