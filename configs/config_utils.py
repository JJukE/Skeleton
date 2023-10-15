from jjuke.network_utils.distributed import is_master_proc
from jjuke.utils.logger import CustomLogger


class CONFIG(object):
    def __init__(self, config):
        self.config = config
        self.is_master = is_master_proc(config.distributed.num_gpus)
        
        self.logger = CustomLogger(log_dir=self.config.log.log_dir, isTrain=self.config.mode == "train")


    def info(self, content):
        if self.config.distributed.num_gpus > 1 and self.is_master:
            self.logger.info(content)
        elif self.config.distributed.num_gpus == 1:
            self.logger.info(content)
    
    def flush(self):
        if self.config.distributed.num_gpus > 1 and self.is_master:
            self.logger.flush()
        elif self.config.distributed.num_gpus == 1:
            self.logger.flush()

