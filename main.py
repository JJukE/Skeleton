import hydra
import os

from jjuke.utils import seed_everything

def single_proc_run(config):
    from configs.config_utils import CONFIG
    from jjuke.network_utils.distributed import initiate_environment
    initiate_environment(config)
    cfg = CONFIG(config)

    if config.mode == 'train':
        from train import Train
        trainer = Train(cfg=cfg)
        trainer.run()
    elif config.mode == 'eval':
        from evaluate import Evaluate
        evaluater = Evaluate(cfg=cfg)
        evaluater.run()
    # elif config.mode == 'interpolation':
    #     from interpolation import Interpolation
    #     interpolater = Interpolation(cfg=cfg)
    #     interpolater.run()
    # elif config.mode == 'generation':
    #     from generation import Generation
    #     generator = Generation(cfg=cfg)
    #     generator.run()
    # elif config.mode == 'demo':
    #     from demo import Demo
    #     example = Demo(cfg=cfg)
    #     example.run()

@hydra.main(config_path='configs/config_files', config_name='example.yaml')
def main(config):
    os.environ["OMP_NUM_THREADS"] = str(config.distributed.OMP_NUM_THREADS)
    # config.root_dir = hydra.utils.get_original_cwd()

    from jjuke.network_utils.distributed import multi_proc_run
    print('Initialize device environments')
    seed_everything(config.seed)
    
    if config.distributed.use_ddp:
        multi_proc_run(config.distributed.num_gpus, func=single_proc_run, func_args=(config,))
    else:
        single_proc_run(config)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()