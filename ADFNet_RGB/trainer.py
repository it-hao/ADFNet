import os
import math
from decimal import Decimal
import numpy as np
import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)):
                self.scheduler.step() 

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (hr, filename, idx_scale) in enumerate(self.loader_train):
            hr = self.prepare([hr])[0]
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            
            noise = torch.FloatTensor(hr.size()).normal_(mean=0, std=self.args.noiseL / 255.).cuda()

            lr = hr + noise
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print("break...")
                exit()
            
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:  
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(  
                    (batch + 1) * self.args.batch_size,  
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                if self.args.test_only:
                    self.ckp.write_log('#####################[dataset={}]---[model={}]---[noise={}]#####################'.format(self.args.testset, self.args.model, str(self.args.noiseL)))
                else:
                    self.ckp.write_log('#####################[dataset={}]---[model={}]---[noise={}]#####################'.format(self.args.data_val, self.args.model, str(self.args.noiseL)))
                avg_psnr = 0
                avg_ssim = 0
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (hr, filename, _) in enumerate(tqdm_test):
                    hr = self.prepare([hr], volatile=True)[0]
                    filename = filename[0]
                    
                    # torch.manual_seed(self.args.seed)
                    noise = torch.FloatTensor(hr.size()).normal_(mean=0, std=self.args.noiseL/255.).cuda()
                    lr = hr + noise

                    sr = self.model(lr, idx_scale)

                    img_sr, img_hr = utility.Tensor2np([sr.data[0].float().cpu(), hr.data[0].float().cpu()], rgb_range=self.args.rgb_range, out_type=np.float64)

                    # 训练时候的验证
                    if not self.args.test_only:
                        tmp_psnr = utility.calc_psnr(img_sr/self.args.rgb_range * 255., img_hr/self.args.rgb_range * 255.)
                        eval_acc += tmp_psnr
                        # print(tmp_psnr)

                    # 只保存 HQ 图像
                    save_list = [utility.quantize(sr, self.args.rgb_range)]

                    # 仅仅是在测试的时候使用，算出来的结果和Matlab代码是一致的
                    if self.args.test_only:
                        psnr, ssim = utility.calc_metrics(img_sr/self.args.rgb_range, img_hr/self.args.rgb_range)
                        avg_psnr += psnr
                        avg_ssim += ssim
                        self.ckp.write_log("{}: psnr = {} ,\tssim = {} .".format(filename, psnr,ssim))

                    ################### 测试和训练的时候只保存DN图片 #####################
                    if self.args.save_results: # 这里训练和测试的时候都会保存图片
                        if self.args.test_only:
                            self.ckp.save_results_test(filename, save_list) 
                        else:
                            self.ckp.save_results(filename, save_list) 

                if self.args.test_only:
                    avg_psnr = avg_psnr / len(self.loader_test)
                    avg_ssim = avg_ssim / len(self.loader_test)
                    # print(avg_psnr, avg_ssim)
                    self.ckp.write_log("{}: psnr = {} ,\tssim = {} .".format('AVG', avg_psnr, avg_ssim))

                if not self.args.test_only:               
                    self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            self.args.data_val,
                            scale,
                            self.ckp.log[-1, idx_scale],
                            best[0][idx_scale],
                            best[1][idx_scale] + 1
                        )
                    )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(idx, tensor):
            if not self.args.cpu: tensor = tensor.cuda()
            if self.args.precision == 'half': tensor = tensor.half()
            # Only test lr can be volatile
            # return Variable(tensor, volatile=(volatile and idx==0))
            return tensor.to(device)
           
        return [_prepare(i, _l) for i, _l in enumerate(l)]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs