import os
from data import srdata

# DIV2k + Flickr2K + BSD400 + WED
class MIX(srdata.SRData):
    def __init__(self, args, train=True):
        super(MIX, self).__init__(args, train)

    def _scan(self):
        list_hr = []

        for hr_apath in self.dir_hr:
            for f in os.listdir(hr_apath):
                try:
                    filename = os.path.join(hr_apath, f)
                    list_hr.append(filename)
                except:
                    pass
        print("The number of datasets is:", len(list_hr))
        return list_hr

    def _set_filesystem(self, dir_data):
        dir_data = dir_data + '/' + self.args.data_train
        # self.dir_hr = [dir_data + '/WED', dir_data + '/FiveK', dir_data + '/DIV2K'] # dir_data + '/Flickr2K'  dir_data + '/BSD400'
        # self.dir_hr = [dir_data + '/WED', dir_data + '/DIV2K', dir_data + '/Flickr2K',  dir_data + '/BSD400', dir_data + '/FiveK'] 
        self.dir_hr = [dir_data + '/WED', dir_data + '/DIV2K', dir_data + '/Flickr2K',  dir_data + '/BSD400']

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * 2       #>>>>>>>>>>>>>>>>>>>>>>>>repeat参数确保重复次数<<<<<<<<<<<<<<<<<<<<<<<<<<<
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

