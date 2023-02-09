import csv
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from algorithm.RoC import Sl_loss
from dataset.idn.data.data_loader import load_noisydata
from dataset.ood.builder import *
from dataset.ccn.data_loader import load_ccn_data


parser = argparse.ArgumentParser()

parser.add_argument('--train_nums', type=int, help='how many times to train', default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, default=None)
parser.add_argument('--batch_size', type=float, default=128)
parser.add_argument('--num_workers', type=float, default=4)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--noise', type=str, help='choose the noise construction from [ccn, ood, idn]', default='idn')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--n_epoch', type=float, default=200)
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--epoch_decay_start', type=int, default=80)

args = parser.parse_args()

class CLDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        return x1

def main():
    print('loading dataset...')

    if args.dataset == 'cifar10':
        input_channel = 3
        num_classes = 10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200

    if args.dataset == 'cifar100':
        input_channel = 3
        num_classes = 100
        args.top_bn = False
        args.epoch_decay_start = 100
        args.n_epoch = 200

    if args.noise == 'idn':
        _, train_loader, _, _, test_loader = load_noisydata(
            dataset=args.dataset,
            noise_type='instance',
            random_state=None,
            batch_size=args.batch_size,
            add_noise=True,
            flip_rate_fixed=args.noise_rate,
            trainval_split=1,
            train_frac=1,
            augment=True
        )
        train_dataset = train_loader.dataset.dataset
        test_dataset = test_loader.dataset
    elif args.noise == 'ood':
        transform = build_transform(rescale_size=32, crop_size=32)
        dataset = build_cifar100n_dataset(os.path.join('./data/'), CLDataTransform(transform['cifar_train']),
                                          transform['cifar_test'],
                                          noise_type=args.noise_type, openset_ratio=0.2, closeset_ratio=args.noise_rate)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    elif args.noise == 'ccn':
        train_loader, test_loader, train_dataset, test_dataset = load_ccn_data(
            dataset=args.dataset,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    print('building model...')

    if args.noise_rate == 0.8:
        args.lr = 0.0001
    else:
        args.lr = 0.01

    model = Sl_loss(args, train_dataset, device, input_channel=input_channel, num_class=num_classes)

    epoch = 0
    acc_list = []

    if not os.path.exists(r'./result'):
        os.makedirs(r'./result')
    fp = 'result/{}-ours-{}-{}-{}.csv'.format(args.noise, args.dataset, args.noise_type, args.noise_rate)
    with open(fp, 'a', newline='') as f:
        row = ['Epoch', 'Train Accuracy', 'Test Accuracy', 'Label Precision', 'Model']
        write = csv.writer(f)
        write.writerow(row)

    test_acc = model.evaluate(test_loader)
    with open(fp, 'a', newline='') as f:
        row = [epoch, 0, test_acc, 0, 'Ours']
        write = csv.writer(f)
        write.writerow(row)

    for epoch in range(0, args.n_epoch):
        # train models
        train_acc1, pure_ratio_1_list = model.train(train_loader, epoch)

        # evaluate models
        test_acc = model.evaluate(test_loader)

        # save results
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f ' % (
                    epoch + 1, args.n_epoch, len(train_dataset), test_acc))
        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f, Pure Ratio 1 %.4f %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc, mean_pure_ratio1))

            with open(fp, 'a', newline='') as f:
                row = [epoch + 1, train_acc1, test_acc, mean_pure_ratio1, 'Ours']
                write = csv.writer(f)
                write.writerow(row)

            if epoch >= 190:
                acc_list.extend([test_acc])

    avg_acc = sum(acc_list) / len(acc_list)
    var_acc = np.var(acc_list)
    print(len(acc_list))
    print("the average acc in last 10 epochs: {}, the variance acc in last 10 epochs: {}".format(str(avg_acc),
                                                                                                 str(var_acc)))

    with open('result/{}-ours-{}-{}-{}-avg.csv'.format(args.noise, args.dataset, args.noise_type, args.noise_rate), 'a',
              newline='') as f:
        row = ['Avg acc', avg_acc, 'Var acc', var_acc]
        write = csv.writer(f)
        write.writerow(row)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('device:{} is used!'.format(device))

    for idx in range(args.train_nums):
        args.result_idx = idx
        print(args)

        main()

