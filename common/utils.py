import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer



def plot_result(accuracy_list,pure_ratio_list,name="test.png"):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_list, label='test_accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(pure_ratio_list, label='test_pure_ratio')
    plt.savefig(name)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    # target = torch.argmax(target,dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


def update_ema_variabels(model, ema_model,  alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_pseudo_label(train_dataset, idx, pseudo_label, beta):
    for i in range(len(idx)):
        train_dataset.targets[idx] = beta * pseudo_label[idx] - (1 - beta) * train_dataset.targets[idx]


def dataloader_to_dict(data, label, index, dict):
    for i in range(data.shape[0]):
        if not index[i].item() in dict.keys():
            dict[index[i].item()] = [data[i], label[i]]


def dict_relabel(dict, index):
    for i in range(index.shape[0]):
        if i == 0:
            label = dict[index[i].item()][1].unsqueeze(0)
        else:
            t = dict[index[i].item()][1].unsqueeze(0)
            label = torch.cat((label, t),dim=0)
    return label

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def to_onehot_label(y):
    y_n = y.numpy()
    lb = LabelBinarizer()
    lb.fit(list(range(0, 10)))
    y_one_hot = lb.transform(y_n)
    floatTensor = torch.FloatTensor(y_one_hot)

    return floatTensor


def softmax_prob(x, beta):
    p = torch.exp(beta * x) / torch.exp(beta * x).sum()
    return p


def str_is_int(x):
    if x.count('-') > 1:
        return False
    if x.isnumeric():
        return True
    if x.startswith('-') and x.replace('-', '').isnumeric():
        return True
    return False


def str_is_float(x):
    if str_is_int(x):
        return False
    try:
        _ = float(x)
        return True
    except ValueError:
        return False


class Config(object):
    def set_item(self, key, value):
        if isinstance(value, str):
            if str_is_int(value):
                value = int(value)
            elif str_is_float(value):
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
        if key.endswith('milestones'):
            try:
                tmp_v = value[1:-1].split(',')
                value = list(map(int, tmp_v))
            except:
                raise AssertionError(f'{key} is: {value}, format not supported!')
        self.__dict__[key] = value

    def __repr__(self):
        # return self.__dict__.__repr__()
        ret = 'Config:\n{\n'
        for k in self.__dict__.keys():
            s = f'    {k}: {self.__dict__[k]}\n'
            ret += s
        ret += '}\n'
        return ret


def load_from_cfg(path):
    cfg = Config()
    if not path.endswith('.cfg'):
        path = path + '.cfg'
    if not os.path.exists(path) and os.path.exists('config' + os.sep + path):
        path = 'config' + os.sep + path
    assert os.path.isfile(path), f'{path} is not a valid config file.'

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    for line in lines:
        if line.startswith('['):
            continue
        k, v = line.replace(' ', '').split('=')
        # if k in supported_fields:
        cfg.set_item(key=k, value=v)
    cfg.set_item(key='cfg_file', value=path)

    return cfg


if __name__ == '__main__':
    x = torch.randn([4, 1])
    print(x)
    print(softmax_prob(x, beta= 0.5))

