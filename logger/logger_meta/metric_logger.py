import matplotlib
import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from .base_logger import BaseLogger
import time

matplotlib.use('Agg')


class MetricLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, cfg):
        super().__init__(tb_logger, log_path, cfg)
        self.NAME = 'metric'
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'batchwise'), exist_ok=True)
        self.phase = None
        self.current_epoch = -1
        self.current_batch = -1
        self.metric_container = dict()
        self.batch_size = self.cfg.BATCHSIZE
        self.epoch_total = self.cfg.EPOCH_TOTAL
        self.phase_time_start = time.time()

    def log_batch(self, batch):
        """
        - add each metric to tensorboard
        - record each metric for epoch save
        - display in terminal, displayed metric is averaged
        """
        if not self.NAME in batch['parser'].keys():
            return
        keys_list = batch['parser'][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch['data']
        total_batch = batch['batch-total']
        self.phase = batch['phase']
        self.current_batch = batch['batch-id']
        self.current_epoch = batch['epoch-id']
        for k in keys_list:
            if not k in self.metric_container.keys():
                self.metric_container[k] = [data[k]]
            else:
                self.metric_container[k].append(data[k])
            # add to tb
            self.tb.add_scalars('Metric-BatchWise/' + k, {self.phase: float(data[k])},
                                self.current_batch + self.current_epoch * total_batch)
        self.update_progbar(self.phase, self.current_epoch, self.current_batch, total_batch,
                            sum(self.metric_container[keys_list[0]]) / len(self.metric_container[keys_list[0]]),
                            (time.time() - self.phase_time_start) / 60)

    def log_phase(self):
        """
        - save the batch-wise scalar in file
        - save corresponding figure for each scalar
        For epoch wise metric
        - add to tb
        """
        phase = self.phase
        batchwise_root = os.path.join(self.log_path, 'batchwise', 'epoch_%d' % self.current_epoch)
        os.makedirs(batchwise_root, exist_ok=True)
        for k, v in self.metric_container.items():
            # first log this phase's batch wise csv
            with open(os.path.join(batchwise_root, '%d_ep_' % self.current_epoch + '_' + k + '_' + phase + '.csv'),
                      'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(v)
            # log figure
            plt.plot(np.array(v))
            plt.title('%d_ep_' % self.current_epoch + k + '_' + phase)
            plt.savefig(os.path.join(batchwise_root, '%d_ep_' % self.current_epoch + '_' + k + '_' + phase + '.png'))
            plt.close()
            # add average to tb epoch wise
            mean = sum(v) / len(v)
            self.tb.add_scalars('Metric-EpochWise/' + k, {phase: float(mean)}, int(self.current_epoch))
        print('\n' + '=' * shutil.get_terminal_size()[0])
        self.metric_container = dict()
        self.phase_time_start = time.time()

    def update_progbar(self, phase, epoch_current, current, total, first_metric, time):
        prefix = '|EP %4d/%4d|SP %4d/%4d|M %.5f|%.1f min|%.1f min est.|' % (
            epoch_current, self.epoch_total, current * self.batch_size, total * self.batch_size,
            first_metric, time, time * (total / current - 1)) + phase + '|'
        termWidth = shutil.get_terminal_size()[0]
        barLength = termWidth - len(prefix) - 14
        print(self.ProgressBar(current / total, prefix, barLength, True), end='  ')

    @classmethod
    def ProgressBar(cls, percent, prefix=None, notches=50, numericalpercent=True, unicode=False):
        """
        https://github.com/paulojraposo/ProgBar
        """
        outString = u""  # Unicode string.
        if prefix:
            prefix = "{} ".format(prefix)
            outString = outString + prefix
        x_of_notches = int(round(percent * notches))
        startCap = "["
        endCap = "]"
        fullSegment = '>'  # '▣'  # "✔"  # ">"    # "#"
        blankSegment = '-'  # '▢'  # "."
        if unicode:
            fullSegment = "\u25AE"  # Full block in Unicode
            blankSegment = "\u25AF"  # Empty block in Unicode
        outString = outString + startCap
        for i in range(x_of_notches):
            outString = outString + fullSegment  # Full block
        for i in range(notches - x_of_notches):
            outString = outString + blankSegment
        outString = outString + endCap
        if numericalpercent:
            outString = outString + " [{}%]".format(str(round(percent * 100, 2)))
        return '\r' + outString
