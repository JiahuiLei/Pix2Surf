"""
excel logger
data structure:
- one head-key is one file
- each passed in data is a dict {col-name:list of values}, each value will be recorded into one row
- there is some basic meta info for each row
"""

import pandas as pd
from .base_logger import BaseLogger
import os


class XLSLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = 'xls'
        os.makedirs(self.log_path, exist_ok=True)
        self.visual_interval_epoch = cfg.VIS_PER_N_EPOCH
        self.record_interval_batch = cfg.VIS_PER_N_BATCH
        self.visual_one = cfg.VIS_ONE_PER_BATCH
        self.visual_train_interval_batch = cfg.VIS_TRAIN_PER_BATCH
        self.pd_container = dict()

        self.current_epoch = 1
        self.current_phase = ''

    def log_batch(self, batch):
        # get data
        if not self.NAME in batch['parser'].keys():
            return
        keys_list = batch['parser'][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch['data']
        phase = batch['phase']
        current_epoch = batch['epoch-id']
        self.current_epoch = current_epoch
        self.current_phase = phase
        meta_info = batch['meta-info']
        # for each key (file)
        for sheet_key in keys_list:
            kdata = data[sheet_key]
            assert isinstance(kdata, dict)
            if sheet_key not in self.pd_container.keys():
                self.pd_container[sheet_key] = pd.DataFrame()
            add_list = list()
            count = len(meta_info['object'])
            for ii in range(count):
                data = dict()
                for k, v in meta_info.items():
                    data[k] = v[ii]
                for k, v in kdata.items():
                    data[k] = v[ii]
                prefix = ""
                for k, v in meta_info.items():
                    prefix += k + "_" + str(v[ii]) + "_"
                data['prefix'] = prefix
                add_list.append(data)
            self.pd_container[sheet_key] = self.pd_container[sheet_key].append(add_list, ignore_index=True)

    def log_phase(self):
        for k in self.pd_container.keys():
            self.pd_container[k].to_excel(
                os.path.join(self.log_path, k + '_' + str(self.current_epoch) + '_' + self.current_phase + '.xls'))
            self.pd_container[k] = pd.DataFrame()
