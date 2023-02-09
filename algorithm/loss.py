import torch
import numpy as np
import torch.nn.functional as F


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def mutual_information_compute(y_1, y_2):
    y_1 = F.softmax(y_1, dim=-1)
    y_2 = F.softmax(y_2, dim=-1)
    return torch.sum((torch.log2((y_1 + y_2)/2) - (torch.log2(y_1) + torch.log2(y_2)) / 2), dim=1)

def loss_drop(y_1, t, forget_rate, ind, noise_or_not):
    loss_pick= F.cross_entropy(y_1, t.long(), reduction='none')
    loss_pick = loss_pick.cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update = ind_sorted[:num_remember]

    # exchange

    loss = torch.mean(loss_pick[ind_update])

    return loss, pure_ratio


def generate_pesudo_label(ema_model,ind_all, dict):
    for i in range(len(ind_all)):
        if i == 0:
            image = dict[ind_all[i].item()][0].unsqueeze(0)
        else:
            t = dict[ind_all[i].item()][0].unsqueeze(0)
            image = torch.cat((image, t), dim=0)
    image = image.cuda()
    pesudo_label = ema_model(image)[:int(0.8 * len(ind_all))].cpu()
    return pesudo_label


class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss
