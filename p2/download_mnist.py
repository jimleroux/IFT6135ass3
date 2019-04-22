import argparse
import gzip
import os
import urllib
import urllib.request

import numpy as np


"""
Code taken from https://github.com/yburda/iwae
============================================================================
"""

def download_bin_mnist():
    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
        local_filename = os.path.join("./dataset", filename)
        urllib.request.urlretrieve(url, local_filename)

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

def binarized_mnist():
    with open("./dataset/binarized_mnist_train.amat") as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open("./dataset/binarized_mnist_train.amat") as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open("./dataset/binarized_mnist_train.amat") as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')

    return train_data, validation_data, test_data

"""
============================================================================
"""

def main(args):
    if args.load_data:
        download_bin_mnist()
    
    train, valid, test = binarized_mnist()

    np.save(args.save_path + "train", train)
    np.save(args.save_path + "valid", valid)
    np.save(args.save_path + "test", test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data',
                        type=bool,
                        default=True,
                        help="Set True if data need to be downloaded.")
    parser.add_argument('--save_path',
                        type=str,
                        default="./dataset/numpy_data/",
                        help="Full path where to save data")
    args = parser.parse_args()
    main(args)
