import os
from data import srdata

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))

        list_hr.sort()

        return list_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmarkColor', self.args.data_val)
        self.dir_hr = os.path.join(self.apath, 'HQ')
        self.ext = '.png' # 图片类型
