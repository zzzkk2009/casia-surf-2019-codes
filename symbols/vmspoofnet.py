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

def height_split(data, num_outputs=2):
    x = mx.sym.split(data, num_outputs=num_outputs, axis=2)
    return x

def width_split(data, num_outputs=2):
    x = mx.sym.split(data, num_outputs=num_outputs, axis=3)
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

def conv(data, num_filter=122, kernel_size=(1, 1), stride=(1,1), pad=(1, 1)):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, pad=pad, kernel=kernel_size, stride=stride)
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')
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
        2.0:(-1, 24, 244, 488, 244, 2048)
    }
    out_channels = width_config[width_multiplier]

    data = mx.sym.var('data')
    data = mx.sym.Convolution(data=data, num_filter=out_channels[1], 
                              kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    # data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max', 
    #                       stride=(2, 2), pad=(1, 1))
    
    data = make_stage(data, out_channels, 2)
    
    data = make_stage(data, out_channels, 3)

    heights = height_split(data)  # (data -> 14x14)
    top = heights[0]     # (7x14)
    # bottom = heights[1]  # (7x14)

    # tops = width_split(top)
    # top_left = tops[0]     # (7x7)
    # top_right = tops[1]    # (7x7)

    # bottoms = width_split(bottom)
    # bottom_left = bottoms[0]     # (7x7)
    # bottom_right = bottoms[1]    # (7x7)

    # widths = width_split(data)
    # left = widths[0]     # (14x7)
    # right = widths[1]    # (14x7)

    data = make_stage(data, out_channels, 4)    # (7x7)
    top = conv(top, num_filter=3*out_channels[4], kernel_size=(3,3), stride=(1,2), pad=(1,1))            # (7x7)
    # sub_num_filter = out_channels[4] // 2
    # top = conv(top, num_filter=sub_num_filter, kernel_size=(1,3), stride=(1,2), pad=(0,1))            # (7x7)
    # bottom = conv(bottom, num_filter=sub_num_filter, kernel_size=(1,3), stride=(1,2), pad=(0,1))      # (7x7)
    # left = conv(left, num_filter=out_channels[4], kernel_size=(3,1), stride=(2,1), pad=(1,0))          # (7x7)
    # right = conv(right, num_filter=out_channels[4], kernel_size=(3,1), stride=(2,1), pad=(1,0))        # (7x7)
    # top_left = conv(top_left, num_filter=out_channels[4], kernel_size=(1,1), stride=(1,1), pad=(0,0))  # (7x7)
    # top_right = conv(top_right, num_filter=out_channels[4], kernel_size=(1,1), stride=(1,1), pad=(0,0))  # (7x7)
    # bottom_left = conv(bottom_left, num_filter=out_channels[4], kernel_size=(1,1), stride=(1,1), pad=(0,0))  # (7x7)
    # bottom_right = conv(bottom_right, num_filter=out_channels[4], kernel_size=(1,1), stride=(1,1), pad=(0,0))  # (7x7)

    # data = mx.sym.concat(top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, data, dim=1)
    # data = mx.sym.concat(top, bottom, left, right, data, dim=1)
    # data = mx.sym.concat(top, bottom, data, dim=1)
    data = mx.sym.concat(top, data, dim=1)

    data = mx.sym.Convolution(data=data, num_filter=out_channels[5], 
                      kernel=(1, 1), stride=(1, 1))
     
    data = mx.sym.Pooling(data=data, kernel=(7, 7), global_pool=True, pool_type='avg')

    # data = mx.sym.Convolution(data=data, num_filter=num_classes, 
    #                   kernel=(1, 1), stride=(1, 1))
    
    data = mx.sym.flatten(data=data)

    data = mx.sym.Dropout(data, p = 0.2)
    
    data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)

    out = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return out