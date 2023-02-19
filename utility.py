import os
import math
import time
import datetime
from functools import reduce
from collections import OrderedDict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
# import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
import imageio
import cv2

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 记录当前时间

        if args.load == '.':
            if args.save == '.': args.save = now  # 保存的文件名
            if args.test_only:
                if args.self_ensemble:
                    self.dir = '../experiment/DN/' + args.save + 'plus'
                else:
                    self.dir = '../experiment/DN/' + args.save
            else:
                self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))


        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        if not args.test_only:
            _make_dir(self.dir + '/model')
            _make_dir(self.dir + '/results')
        else:
            _make_dir(self.dir + '/' + self.args.testset  + '/N' + str(self.args.noiseL))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        # //////////////////////////////////////////////////////////
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
        # //////////////////////////////////////////////////////////

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_val)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_val))
        plt.close(fig)

    def save_results_test(self, filename, save_list):
        if self.args.self_ensemble:
            filename = filename.split('.')[0] + '_' + self.args.model + '_plus_N' + str(self.args.noiseL)
        else:
            filename = filename.split('.')[0] + '_' + self.args.model + '_N' + str(self.args.noiseL) # experiment/DN/Kodak24/N10   /kodak01_RDAN_N10_DN.png
        filename = '{}/{}/N{}/{}'.format(self.dir, self.args.testset, str(self.args.noiseL), filename)
        postfix = ('DN', 'LQ', 'HQ')
        # for v, p in zip(save_list, postfix):
        #     tu.save_image(v.data[0] / self.args.rgb_range, '{}.png'.format(filename))
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy().squeeze()
            imageio.imsave('{}_{}.png'.format(filename, p), ndarr)

    def save_results(self, filename, save_list):  # experiment/'model'/results/kodim01_model_N10.png
        filename = filename.split('.')[0] + '_' + self.args.model + '_N' + str(self.args.noiseL)
        filename = '{}/results/{}'.format(self.dir, filename)
        postfix = ('DN', 'LQ', 'HQ')
        # for v, p in zip(save_list, postfix):
        #     print(v.shape)
        #     tu.save_image(v.data[0] / self.args.rgb_range, '{}{}.png'.format(filename, p))
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy().squeeze()  #因为misc.imsave保存图片有三种形式，灰度土图片只能似乎H*W格式，压缩维度
            imageio.imsave('{}_{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def Tensor2np(tensor_list, rgb_range, out_type=np.uint8):
    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0))
        if out_type == np.uint8:
            return array.astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]

################################################################################
                #              calc_metrics            #
################################################################################
def calc_metrics(img1, img2):
    psnr = calc_psnr(img1 * 255, img2 * 255)
    ssim = calc_ssim(img1 * 255, img2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())  # 只优化需要反向传播的

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':  # default: 'step'
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,  # 每过lr_deacay=200次进行一次学习率衰减
            gamma=args.gamma  # gamma = 0.5
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    # if scheduler.last_epoch == -1 and args.start_epoch != 1:
    # scheduler.last_epoch = args.start_epoch
    return scheduler

