# -*- coding:utf-8 -*-

import mxnet as mx
import numpy as np
import time
from collections import namedtuple
import argparse

# namedtuple('名称', [属性list])
Batch = namedtuple('Batch', ['data'])

def get_context():
    return mx.cpu()

def get_mx_input(fname):
    img = mx.image.imread(fname)
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 112, 112)
    img = img.astype('float32') # fix bug: Incompatible attr in node  at 1-th input: expected uint8, got float32
    # print(img.dtype)
    # img = img / 255
    # img = mx.image.color_normalize(img,
    #                 mean=mx.nd.array([0.485, 0.456, 0.406]),
    #                 std=mx.nd.array([0.229, 0.224, 0.225]))
    img = mx.image.color_normalize(img, mean=mx.nd.array([123.68,116.779,103.939]))
    img = img.transpose((2, 0, 1)) # channel first
    img = img.expand_dims(axis=0) # batchify
    # img = img.astype('float32') # for gpu context
    return img

def load_model(prefix, epoch):
    #_prefix = 'checkpoint/checkpoint'
    #_epoch = 517
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # print(sym)
    # print(arg_params)
    # print(aux_params)
    mod = mx.mod.Module(symbol=sym, context=get_context(), data_names=['data'], label_names=None) # label_names can be empty
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))], label_shapes=mod._label_shapes)
    # fix bug: RuntimeError: softmax_label is not presented
    arg_params['softmax_label'] = mx.nd.array([0])
    mod.set_params(arg_params, aux_params)
    return mod

def predict(mod, img_path, type):
    img = get_mx_input(img_path)
    # img = single_input(img_path)
    mod.forward(Batch([img]))
    # internals = mod.symbol.get_internals()
    # print(internals.list_outputs())
    # print('mod.get_outputs() length=', len(mod.get_outputs())) # length = 1
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob) # [[]] -> [] 转为秩为1的矩阵
    # squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    sort_prob_idx = np.argsort(prob)[::1]
    for i in sort_prob_idx:
        print('img_path=%s, probability=%f, class=%s, type=%s' %(img_path, prob[i], i, type))
    return prob

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='test')
    parser.add_argument('filename', help='path to test list file.')
    parser.add_argument('--load-epoch', default=73,
        help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()
    return args

if __name__ == '__main__': 

    time_start = time.time()
    args = parse_args()
    prefix_depth = 'checkpoint_depth_112_29266_38208_vmspoofnet_2m/checkpoint_depth_112_29266_38208_vmspoofnet_2m'
    # epoch_depth = 554
    epoch_depth = int(args.load_epoch)

    mod_depth = load_model(prefix_depth, epoch_depth)

    #time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    date = time.strftime("%Y-%m-%d", time.localtime()) 

    test_list_filename = args.filename # '../phase2/test_public_list.txt'
    with open(test_list_filename, 'r') as f:
        with open('commit_phase2_depth_{}_server_{}.txt'.format(date, epoch_depth), 'w') as df:
            for line in f.readlines():
                line = line.strip() # 去掉每行头尾空白
                line_lst = line.split() # 按空白符分割
                # color_path = line_lst[0]
                depth_path = line_lst[1]
                # ir_path = line_lst[2]

                time0 = time.time()
                depth_prob = predict(mod_depth, '../phase2/' + depth_path, 'depth')
                # print(depth_prob)
                time1 = time.time()
                print('predict time={0}'.format(time1 - time0))

                line_depth = line + ' ' + str(round(depth_prob[-1], 8)) + '\n'
                # print(line_depth)
                df.writelines(line_depth)

    time_end = time.time()
    print('test total time={0}'.format(time_end - time_start))

                






