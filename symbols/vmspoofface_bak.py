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

    # stage_repeats = [3, 7, 3] # 4, 8, 4
    stage_repeats = [4, 8, 4]

    data = shuffleUnit(data, out_channels[stage - 1], out_channels[stage], 
                       'concat', groups, 2, 2)
    
    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnit(data, out_channels[stage], out_channels[stage], 
                           'concat', groups, 1, 1)

    return data

def get_symbol(num_classes, **kwargs):

    width_multiplier = 3.0
    width_config = {
        0.5:(-1, 24, 48, 96, 192, 1024),
        1.0:(-1, 24, 116, 232, 464, 1024),
        1.5:(-1, 24, 176, 352, 704, 1024),
        2.0:(-1, 24, 360, 488, 976, 2048),
        3.0:(-1, 48, 720, 976, 1464, 2048)
    }
    out_channels = width_config[width_multiplier]

    data = mx.sym.var('data') # 112x112

    #(height,width)
    heights = height_split(data, 4)    # 28x112
    row1s = width_split(heights[0], 4) # 28x28
    row2s = width_split(heights[1], 4) # 28x28
    row3s = width_split(heights[2], 4) # 28x28
    row4s = width_split(heights[3], 4) # 28x28

    center_top_1 = mx.sym.concat(row1s[1], row1s[2], dim=3) # 28x56
    center_top_2 = mx.sym.concat(row2s[1], row2s[2], dim=3) # 28x56
    center_top_3 = mx.sym.concat(row3s[1], row3s[2], dim=3) # 28x56
    center_top_4 = mx.sym.concat(row4s[1], row4s[2], dim=3) # 28x56

    center_left_1 = mx.sym.concat(row1s[0], row1s[1], dim=3) # 28x56
    center_left_2 = mx.sym.concat(row2s[0], row2s[1], dim=3) # 28x56
    center_left_3 = mx.sym.concat(row3s[0], row3s[1], dim=3) # 28x56
    center_left_4 = mx.sym.concat(row4s[0], row4s[1], dim=3) # 28x56

    center_right_1 = mx.sym.concat(row1s[2], row1s[3], dim=3) # 28x56
    center_right_2 = mx.sym.concat(row2s[2], row2s[3], dim=3) # 28x56
    center_right_3 = mx.sym.concat(row3s[2], row3s[3], dim=3) # 28x56
    center_right_4 = mx.sym.concat(row4s[2], row4s[3], dim=3) # 28x56

    center = mx.sym.concat(center_top_2, center_top_3, dim=2) # 56x56
    center_top = mx.sym.concat(center_top_1, center_top_2, dim=2) # 56x56
    center_right = mx.sym.concat(center_right_2, center_right_3, dim=2) # 56x56
    center_bottom = mx.sym.concat(center_top_3, center_top_4, dim=2) # 56x56
    center_left = mx.sym.concat(center_left_2, center_left_3, dim=2) # 56x56

    top_left = mx.sym.concat(center_left_1, center_left_2, dim=2) # 56x56
    top_right = mx.sym.concat(center_right_1, center_right_2, dim=2) # 56x56
    bottom_left = mx.sym.concat(center_left_3, center_left_4, dim=2) # 56x56
    bottom_right = mx.sym.concat(center_right_3, center_right_4, dim=2) # 56x56

    center = shuffleUnit(center, 3, out_channels[1]*4, 'concat', 2, 2, 2) # 28x28
    center_top = shuffleUnit(center_top, 3, out_channels[1]*4, 'concat', 2, 2, 2) # 28x28                       
    center_right = shuffleUnit(center_right, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28
    center_bottom = shuffleUnit(center_bottom, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28
    center_left = shuffleUnit(center_left, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28

    top_left = shuffleUnit(top_left, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28
    top_right = shuffleUnit(top_right, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28
    bottom_left = shuffleUnit(bottom_left, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28
    bottom_right = shuffleUnit(bottom_right, 3, out_channels[1], 'concat', 2, 2, 2) # 28x28

    # c22 = conv(row2s[1], num_filter=out_channels[1]*3, kernel_size=(1,1), stride=(1,1), pad=(0,0)) # 28x28
    # c23 = conv(row2s[2], num_filter=out_channels[1]*3, kernel_size=(1,1), stride=(1,1), pad=(0,0)) # 28x28
    # c32 = conv(row3s[1], num_filter=out_channels[1], kernel_size=(1,1), stride=(1,1), pad=(0,0)) # 28x28
    # c33 = conv(row3s[2], num_filter=out_channels[1], kernel_size=(1,1), stride=(1,1), pad=(0,0)) # 28x28

    # data = mx.sym.Convolution(data=data, num_filter=out_channels[1]*2, 
    #                           kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    # data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max', 
    #                       stride=(2, 2), pad=(1, 1))
    # data = mx.sym.concat(center, center_top, center_right, center_bottom, center_left, 
    #     c22, c23, c32, c33, data, dim=1) # 28x28

    # data = mx.sym.concat(center, center_top, center_right, center_bottom, center_left, 
    #     c22, c23, c32, c33, dim=1) # 28x28

    data = mx.sym.concat(center, center_top, center_right, center_bottom, center_left,
        top_left, top_right, bottom_left, bottom_right, dim=1) # 28x28
    
    data = make_stage(data, out_channels, 3) # 14x14
    data = make_stage(data, out_channels, 4) # (7x7)

    data = mx.sym.Convolution(data=data, num_filter=out_channels[5], 
                      kernel=(1, 1), stride=(1, 1))
     
    data = mx.sym.Pooling(data=data, kernel=(7, 7), global_pool=True, pool_type='avg')

    data = mx.sym.Convolution(data=data, num_filter=num_classes, 
                      kernel=(1, 1), stride=(1, 1))
    
    data = mx.sym.flatten(data=data)
    
    # data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)
    
    out = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return out