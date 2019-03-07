# -*- coding:utf-8 -*-

import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
import mxnet as mx

def get_nir_flow_data():
    data_dir="./data/"
    fnames = (os.path.join(data_dir, "train_depth_all_112_29266.rec"),
              os.path.join(data_dir, "val_depth_all_112_9608.rec"))
    return fnames

 # 样本类别不均衡：如果每个分类的样例数量与其他类别数量差距太大，则模型可能倾向于数量占主导地位的类，因为它会让错误率变低。
if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = get_nir_flow_data()

    # parse args
    parser = argparse.ArgumentParser(description="train-casia-surf-depth",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 101,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 2,
        num_examples   = 29266,  # 训练样本数
        image_shape    = '3,112,112', # channel,height,width
        pad_size       = 0,

        # data aug
        max_random_rotate_angle = 45,
        max_random_aspect_ratio = 0.5,
        max_random_shear_ratio = 0.5,
        max_random_h = 15,
        max_random_s = 15,
        max_random_l = 15,
        # max_random_scale = 0,
        # min_random_scale = 0,
        # random_crop = 0,
        # train
        batch_size     = 512,
        num_epochs     = 1000,
        #wd             = 0.000001,
        lr             = 1e-1,
        #lr_factor      = 0.5,
        lr_step_epochs = '10,30,50',
        model_prefix   = 'checkpoint_depth_112_29266_resnet101',
        checkpoint_period = 10, # How many epochs to wait before checkpointing. Defaults to 1.
	    #load_epoch     = 60,
	    gpus           = '1,2,3'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)

