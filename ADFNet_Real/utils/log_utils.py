import logging
import os
import os.path as osp

def logger(name, filepath):
    dir_path = osp.dirname(filepath)
    if not osp.exists(dir_path):
        os.mkdir(dir_path)

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s |[%(lineno)03d]%(filename)-11s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(formatter)
    lg.addHandler(stream_hd)

    file_hd = logging.FileHandler(filepath)
    file_hd.setFormatter(formatter)
    lg.addHandler(file_hd)

    return lg