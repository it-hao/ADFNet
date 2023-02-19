import os
from data import common
import imageio
import torch.utils.data as data
# 测试的时候输入HQ图片，直接进行测试PSNR和SSIM的值

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = False
        self.noise_level = args.noiseL

        hr_apath = args.hrpath + '/' + args.testset # testsets/Set12

        self.hr_filelist = []
        if not train:
            for f in os.listdir(hr_apath):
                try:
                    filename = os.path.join(hr_apath, f)
                    self.hr_filelist.append(filename)
                except:
                    pass
            self.hr_filelist.sort()  

    def __getitem__(self, idx):
        filename = os.path.split(self.hr_filelist[idx])[-1] # 0001.png
        filename, _ = os.path.splitext(filename) # 0001

        hr = imageio.imread(self.hr_filelist[idx])
        hr = common.set_channel([hr], self.args.n_colors)[0]
        hr_tensor = common.np2Tensor([hr], self.args.rgb_range)[0]
        return hr_tensor, filename

    def __len__(self):
        return len(self.hr_filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
