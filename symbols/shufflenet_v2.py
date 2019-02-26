# -*- coding:utf-8 -*-

import mxnet as mx

def combine(residual, data, combine):
    if combine == 'add':
        return residual + data
    elif combine == 'concat':
        return mx.sym.concat(residual, data, dim=1)
    return None

def channel_split(data, groups):
    x = mx.sym.split(data, num_outputs=groups, axis=1)
    return x

# reshape函数shape param values:
#  0: 直接将对应位置input的shape值，拷贝到output的shape位置
#       input shape=(2,3,4),  shape=(4,0,2), output shape=(4,3,2)
#  -1: 利用余下的input的shape，推断output的shape维度
#       input shape=(2,3,4),  shape=(6,1,-1), output shape=(6,1,4)
#  -2：将余下input的shape维度，拷贝到output的shape维度
#       input shape=(2,3,4),  shape=(-2,1,1), output shape=(2,3,4,1,1)
#  -3：使用input的shape连续两位的维度值得乘积，作为output的shape维度值
#       input shape=(2,3,4,5),  shape=(-3,-3), output shape=(6,20)
#  -4：将input的shape中的维度值，分解为shape参数中-4后面连续两位的维度值（可以包含-1）
#       input shape=(3,6,4), shape=(3,-4,2,3,4), output shape=(3,2,3,4)
def channel_shuffle(data, groups):
    data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2)) # shape:(batch_size,groups,height,width)
    data = mx.sym.swapaxes(data, 1, 2)
    data = mx.sym.reshape(data, shape=(0, -3, -2))
    return data

def shuffleUnit(residual, in_channels, out_channels, combine_type='concat', groups=2, DWConv_stride=2, unit_type=1):

    if unit_type == 1:
        x = channel_split(residual, groups)
        residual = x[0]
        data = x[1]
    else:
        data = residual

    split_channel = out_channels // groups

    data = mx.sym.Convolution(data=data, num_filter=split_channel, 
                      kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    data = mx.sym.Convolution(data=data, num_filter=split_channel, kernel=(3, 3), 
                       pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=split_channel)
    data = mx.sym.BatchNorm(data=data)

    data = mx.sym.Convolution(data=data, num_filter=split_channel, 
                       kernel=(1, 1), stride=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    if unit_type == 2:
        residual = mx.sym.Convolution(data=residual, num_filter=in_channels, kernel=(3, 3), 
                       pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=in_channels)
        residual = mx.sym.BatchNorm(data=residual)
        residual = mx.sym.Convolution(data=residual, num_filter=split_channel, 
                      kernel=(1, 1), stride=(1, 1))
        residual = mx.sym.BatchNorm(data=residual)
        residual = mx.sym.Activation(data=residual, act_type='relu')

    data = combine(residual, data, combine_type)

    data = channel_shuffle(data, groups)

    return data

def make_stage(data, out_channels, stage, groups=2):

    stage_repeats = [3, 7, 3] # 4, 8, 4

    data = shuffleUnit(data, out_channels[stage - 1], out_channels[stage], 
                       'concat', groups, 2, 2)
    
    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnit(data, out_channels[stage], out_channels[stage], 
                           'concat', groups, 1, 1)

    return data

def get_symbol(num_classes, **kwargs):

    width_multiplier = 2.0
    width_config = {
        0.5:(-1, 24, 48, 96, 192, 1024),
        1.0:(-1, 24, 116, 232, 464, 1024),
        1.5:(-1, 24, 176, 352, 704, 1024),
        2.0:(-1, 24, 244, 488, 976, 2048)
    }
    out_channels = width_config[width_multiplier]

    data = mx.sym.var('data')
    data = mx.sym.Convolution(data=data, num_filter=out_channels[1], 
                              kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max', 
                          stride=(2, 2), pad=(1, 1))
    
    data = make_stage(data, out_channels, 2)
    
    data = make_stage(data, out_channels, 3)
    
    data = make_stage(data, out_channels, 4)

    data = mx.sym.Convolution(data=data, num_filter=out_channels[5], 
                      kernel=(1, 1), stride=(1, 1))
     
    data = mx.sym.Pooling(data=data, kernel=(7, 7), global_pool=True, pool_type='avg')
    
    data = mx.sym.flatten(data=data)
    
    data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)
    
    out = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return out