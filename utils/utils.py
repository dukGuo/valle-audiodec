import yaml
import torch
import collections
import os
import numpy as np
import random

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Unsupported value encountered.')
 

# def hparams_string(hparams, args):
#     print('Output path: {}'.format(args.logdir_path))
#     values = hparams.values()
#     hp = ['  %s: %s' % (name, values[name])
#           for name in sorted(values) if name != 'sentences']
#     # print('Hyperparameters:\n' + '\n'.join(hp))
#     return


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_config_from_file(file):
    with open(file, 'r') as f:
        hp = yaml.load(f,Loader=yaml.FullLoader)
    hp = HParams(**hp)
    return hp

def update_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr
    return

def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device

def calculate_model_params(model):
    total = sum([param.numel() for param in model.parameters()])
    para_name = [para[0] for para in list(model.named_parameters())]
    print("==================================================")
    print("model struct is {}\n".format(str(model)))
    print("model params : {}".format(para_name))
    # log("FLOPs: {}".format(flops.total_float_ops))
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("==================================================")
    return

def get_metadata(path):
    with open(path, 'r') as f:
        metas = [l.strip() for l in f]
    random.shuffle(metas)
    return metas

class TemperatureSampler():
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        # Create a categorical distribution with temperature adjusted logits
        dist = torch.distributions.categorical.Categorical(logits=logits / self.temperature)

        # Sample
        return dist.sample()



class TopKSampler():
    """
    ## Top-k Sampler
    """
    def __init__(self, k: int, sampler=TemperatureSampler()):
        """
        :param k: is the number of tokens to pick
        :param sampler: is the sampler to use for the top-k tokens

        `sampler` can be any sampler that takes a logits tensor as input and returns a token tensor;
         e.g. [`TemperatureSampler'](temperature.html).
        """
        self.k = k
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """
        # New logits filled with $-\infty$; i.e. zero probability
        zeros = logits.new_ones(logits.shape) * float('-inf')
        # Pick the largest $k$ logits and their indices
        values, indices = torch.topk(logits, self.k, dim=-1)
        # Set the values of the top-k selected indices to actual logits.
        # Logits of other tokens remain $-\infty$
        zeros.scatter_(-1, indices, values)

        # Sample from the top-k logits with the specified sampler.
        return self.sampler(zeros)