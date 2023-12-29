import os
from .dataset_rgb import DataLoaderTrain, DataLoaderVal, DataLoaderTest#, DataLoaderTestSR

# def get_training_data(rgb_dirs, img_options):
#     for rgb_dir in rgb_dirs:
#         assert os.path.exists(rgb_dir) 
#     return DataLoaderTrain(rgb_dirs, img_options, None)

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


# def get_test_data_SR(rgb_dir):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTestSR(rgb_dir, None)


