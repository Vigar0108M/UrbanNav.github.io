import os
import json
import numpy as np
from statistics import mean
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(
            self, 
            mode: str,
            result_folder: str,
            world_size: int,
            iter_log_freq: int = 100,
            
            ):

        self.mode = mode
        self.result_folder = result_folder
        self.world_size = world_size
        self.iter_log_freq = iter_log_freq
        self.cur_data = {}
        self.cur_epoch = 0
        self.train_datas = []
        self.eval_datas = []

        self.loss_file = os.path.join(self.result_folder, f'{self.mode}_loss.jsonl')
        self.writer = SummaryWriter(os.path.join(self.result_folder, 'runs'))
    

    def _write(self, tag: str):

        epoch_loss_dict = {}
        epoch_loss_dict["epoch"] = self.cur_epoch

        for loss_name, iter_losses in self.cur_data.items():
            epoch_loss = mean(iter_losses)
            self.writer.add_scalar(f"{tag}/epoch_{loss_name}", epoch_loss, self.cur_epoch)
            epoch_loss_dict[f"{tag}_{loss_name}"] = epoch_loss

        with open(self.loss_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(epoch_loss_dict, ensure_ascii=False) + '\n')


    def update(
            self,
            tag: str,
            epoch: int,
            data: dict
    ):
        self.cur_data = data
        self.cur_epoch = epoch
        self._write(tag)
        
        