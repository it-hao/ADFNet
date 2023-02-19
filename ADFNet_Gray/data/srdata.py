import os
import numpy as np
import imageio
import torch
import torch.utils.data as data
import cv2

from data import common
'''
只输入HQ 图像,LQ图像直接在HQ图像上加入噪声
'''
class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.noise_level = args.noiseL
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())

        if args.ext == 'img' or benchmark:   
            self.images_hr = self._scan()

        elif args.ext.find('sep') >= 0:
            self.images_hr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    # hr = imageio.imread(v)
                    img_color = cv2.imread(v)
                    hr = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
              
            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        hr = self._get_patch(hr)
        hr = common.set_channel([hr], self.args.n_colors)[0]
        hr_tensor = common.np2Tensor([hr], self.args.rgb_range)[0]
        return hr_tensor, filename
        
    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            # hr = imageio.imread(hr)
            # cv2.imread(hr, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(hr)
            hr = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return hr, filename

    def _get_patch(self, hr):
        patch_size = self.args.patch_size

        if self.train:
            hr = common.get_patch(hr, patch_size)
            hr = common.augment([hr])[0]

        return hr 

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

