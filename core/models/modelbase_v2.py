"""
Model base Version 2
"""
import torch.nn as nn
from torch.nn import init
import torch


class ModelBase(object):
    def __init__(self, cfg):
        """
        Key components:
            gpu_list; gpu_list from cfg
            optimizer
            network: one network in on model, compute loss inside each network on different gpus
        Key functions need to implement:
            _preprocess
            _predict
            _postprocess
            train/vali_batch: composed of the three methods above
        """
        self.model_name = 'Model Base'
        # init device
        self.gpu_list = cfg.GPU
        self.__dataparallel_flag__ = False  # flag: whether use DataParallel
        # init optimizer
        self.lr = cfg.LR
        self.optimizer = None
        # init network
        self.network = None
        # clarify output meanings
        self.output_info_dict = {
            'metric': list(),
        }

    # models core methods #########################################################
    def _preprocess(self, batch):
        """
        get a batch from dataset.__getitem__ following a collate_fn
        Pass batch dict from CPU to GPU
        """
        device = torch.device("cuda")
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        return batch

    def _predict(self, batch, is_train=True):
        """
        forward through the network,
        """
        nn_out = self.network(batch['to-nn'], is_train=is_train)
        for k, v in nn_out.items():
            batch[k] = v
        return batch

    def _postprocess(self, batch):
        """
        Post process, get multi GPU batch dicts, process on one gpu or cpu
        :return: a dictionary
        """
        return batch

    def train_batch(self, batch):
        batch = self._preprocess(batch)
        self.set_train()
        self.zero_grad()
        batch = self._predict(batch)
        batch = self._postprocess(batch)
        if self.__dataparallel_flag__:
            for k in batch.keys():
                if k.endswith('loss') or k in self.output_info_dict['metric']:
                    if isinstance(batch[k], list):
                        for idx in len(batch[k]):
                            batch[k][idx] = batch[k][idx].mean()
                    else:
                        batch[k] = batch[k].mean()
        batch['batch-loss'].backward()
        self.optimizer.step()
        return batch

    def vali_batch(self, batch):
        batch = self._preprocess(batch)
        self.set_eval()
        with torch.no_grad():
            batch = self._predict(batch, is_train=False)
        batch = self._postprocess(batch)
        if self.__dataparallel_flag__:
            for k in batch.keys():
                if k.endswith('loss') or k in self.output_info_dict['metric']:
                    if isinstance(batch[k], list):
                        for idx in len(batch[k]):
                            batch[k][idx] = batch[k][idx].mean()
                    else:
                        batch[k] = batch[k].mean()
        return batch

    def test_batch(self, batch):
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    # models I/O methods ##########################################################
    # load, save, init refers to https://github.com/xiumingzhang/GenRe-ShapeHD
    def load_model(self, loadpath, current_model_state='cpu', load_optimizer=False, strict=True):
        assert current_model_state in ['cpu'], "Model Loading Error!"
        print("Load Model from: ")
        print(loadpath)
        device = torch.device(current_model_state)
        if isinstance(loadpath, list):
            for path in loadpath:
                checkpoint = torch.load(path, map_location=device)
                if self.__dataparallel_flag__:
                    for k in self.network.module.network_dict.keys():
                        if k in checkpoint.keys():
                            self.network.module.network_dict[k].load_state_dict(checkpoint[k], strict=strict)
                    if load_optimizer:
                        self.optimizer.module.load_state_dict(checkpoint['optimizer'], strict=strict)
                else:
                    for k in self.network.network_dict.keys():
                        if k in checkpoint.keys():
                            self.network.network_dict[k].load_state_dict(checkpoint[k], strict=strict)
                    if load_optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer'], strict=strict)
        else:
            path = loadpath
            checkpoint = torch.load(path, map_location=device)
            if self.__dataparallel_flag__:
                for k in self.network.module.network_dict.keys():
                    if k in checkpoint.keys():
                        self.network.module.network_dict[k].load_state_dict(checkpoint[k])
                if load_optimizer:
                    self.optimizer.module.load_state_dict(checkpoint['optimizer'])
            else:
                for k in self.network.network_dict.keys():
                    if k in checkpoint.keys():
                        self.network.network_dict[k].load_state_dict(checkpoint[k])
                if load_optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint

    def save_model(self, filepath, additional_dict=None):
        save_dict = {}
        # add additional info
        if additional_dict is not None:
            for k, v in additional_dict.items():
                save_dict[k] = v
        # save all self.__networks_dict param
        if self.__dataparallel_flag__:
            for net_name in self.network.module.network_dict.keys():
                save_dict[net_name] = self.network.module.network_dict[net_name].state_dict()
        else:
            for net_name in self.network.network_dict.keys():
                save_dict[net_name] = self.network.network_dict[net_name].state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        # save
        torch.save(save_dict, filepath)

    def init_weight(self, net=None, init_type='kaiming', init_param=0.02):
        """
        Use nn.Module.apply(fn) to recursively apply initialization to the Model
        If pass in a network, init the passed in net
        else, init self.net
        """

        def init_func(m, init_type=init_type):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orth':
                    init.orthogonal_(m.weight.data, gain=init_param)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                try:
                    init.normal_(m.weight.data, 1.0, init_param)
                    init.constant_(m.bias.data, 0.0)
                except:
                    try:
                        init.normal_(m.bn.weight.data, 1.0, init_param)
                        init.constant_(m.bn.bias.data, 0.0)
                    except:
                        raise ValueError("Can't Initialize BN")

        if net is not None:
            net.apply(init_func)
        else:
            self.network.apply(init_func)

    # models state methods ########################################################

    def to_gpus(self):
        # set device (if no gpu is available or gpu=[-1])
        device = torch.device("cuda")
        if len(self.gpu_list) > 1:
            self.network = nn.DataParallel(self.network)
            self.__dataparallel_flag__ = True
        self.network.to(device)

    def set_train(self):
        """
        Set models's network to train mode
        """
        self.network.train()

    def set_eval(self):
        """
        Set models's network to eval mode
        WARNING! THIS METHOD IS VERY IMPORTANT DURING VALIDATION BECAUSE OF PYTORCH BN MECHANISM
        """
        self.network.eval()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network_dict = None

    def forward(self, *input):
        raise NotImplementedError
