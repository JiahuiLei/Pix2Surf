"""
Wrapper
"""


class Trainer(object):
    def __init__(self, cfg, model, dataloader_dict, logger):
        self.cfg = cfg
        self.modes = self.cfg.MODES
        self.model = model
        self.dataloader_dict = dataloader_dict
        self.logger = logger
        self.current_epoch = cfg.RESUME_EPOCH_ID if cfg.RESUME else 1
        self.batch_count = 0

    def do_epoch(self):
        # for multi phase of one epoch
        for mode in self.modes:
            if mode in self.dataloader_dict.keys():
                # start this phase, pass forward all samples in data-set
                self.batch_count = 0
                batch_total_num = len(self.dataloader_dict[mode])
                for batch in iter(self.dataloader_dict[mode]):
                    self.batch_count += 1
                    if mode == 'train':
                        batch = self.model.train_batch(batch)  # actual training line
                    else:
                        batch = self.model.vali_batch(batch)  # actual validation/test line
                    batch = self.wrap_batchwise_log(batch, batch_total_num, mode=mode, )
                    self.logger.log(batch)
        self.current_epoch += 1
        self.adjust_learning_rate(optimizer=self.model.optimizer, epoch=self.current_epoch,
                                  lr_decay_rate=self.cfg.OPTI_DECAY_RATE,
                                  lr_decay_epoch=self.cfg.OPTI_DECAY_INTERVAL, min_lr=self.cfg.OPTI_DECAY_MIN)
        if self.current_epoch == self.cfg.EPOCH_TOTAL + 1:  # if the last epoch
            self.logger.end_log()
        return self.current_epoch

    def wrap_batchwise_log(self, batch, batch_total, mode='train'):
        try:
            assert 'meta-info' in batch.keys()  # Valid batch must have a key called meta-info
        except:
            raise ValueError("A valid batch passing to logger must contain a member called 'meta-info'!")
        wrapped = dict()
        # add logger head
        wrapped['batch-id'] = self.batch_count
        wrapped['batch-total'] = batch_total
        wrapped['epoch-id'] = self.current_epoch
        wrapped['phase'] = mode.lower()
        # add parser
        wrapped['parser'] = self.model.output_info_dict
        # add save method
        wrapped['save-method'] = self.model.save_model
        # add meta info
        wrapped['meta-info'] = batch['meta-info']
        # main data #################################################
        wrapped['data'] = batch
        return wrapped

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
        if ((epoch + 1) % lr_decay_epoch) != 0:
            return

        for param_group in optimizer.param_groups:
            # print(param_group)
            lr_before = param_group['lr']
            param_group['lr'] = param_group['lr'] * lr_decay_rate
            param_group['lr'] = max(param_group['lr'], min_lr)
        print('changing learning rate {:5f} to {:.5f}'.format(lr_before, max(param_group['lr'], min_lr)))

    @staticmethod
    def reset_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            # print(param_group)
            lr_before = param_group['lr']
            param_group['lr'] = lr
        print('changing learning rate {:5f} to {:.5f}'.format(lr_before, lr))
