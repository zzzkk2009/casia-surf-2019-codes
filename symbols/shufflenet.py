import mxnet as mx

def combine(residual, data, combine):
    if combine == 'add':
        return residual + data
    elif combine == 'concat':
        return mx.sym.concat(residual, data, dim=1)
    return None

def channel_shuffle(data, groups):
    data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
    data = mx.sym.swapaxes(data, 1, 2)
    data = mx.sym.reshape(data, shape=(0, -3, -2))
    return data

def shuffleUnit(residual, in_channels, out_channels, combine_type, groups=3, grouped_conv=True):

    if combine_type == 'add':
        DWConv_stride = 1
    elif combine_type == 'concat':
        DWConv_stride = 2
        out_channels -= in_channels

    first_groups = groups if grouped_conv else 1

    bottleneck_channels = out_channels // 4

    data = mx.sym.Convolution(data=residual, num_filter=bottleneck_channels, 
                      kernel=(1, 1), stride=(1, 1), num_group=first_groups)
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    data = channel_shuffle(data, groups)

    data = mx.sym.Convolution(data=data, num_filter=bottleneck_channels, kernel=(3, 3), 
                       pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=groups)
    data = mx.sym.BatchNorm(data=data)

    data = mx.sym.Convolution(data=data, num_filter=out_channels, 
                       kernel=(1, 1), stride=(1, 1), num_group=groups)
    data = mx.sym.BatchNorm(data=data)

    if combine_type == 'concat':
        residual = mx.sym.Pooling(data=residual, kernel=(3, 3), pool_type='avg', 
                              stride=(2, 2), pad=(1, 1))

    data = combine(residual, data, combine_type)

    return data

def make_stage(data, stage, groups=3):
    stage_repeats = [3, 7, 3]

    grouped_conv = stage > 2

    if groups == 1:
        out_channels = [-1, 24, 144, 288, 567]
    elif groups == 2:
        out_channels = [-1, 24, 200, 400, 800]
    elif groups == 3:
        out_channels = [-1, 24, 240, 480, 960]
    elif groups == 4:
        out_channels = [-1, 24, 272, 544, 1088]
    elif groups == 8:
        out_channels = [-1, 24, 384, 768, 1536]
       
    data = shuffleUnit(data, out_channels[stage - 1], out_channels[stage], 
                       'concat', groups, grouped_conv)

    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnit(data, out_channels[stage], out_channels[stage], 
                           'add', groups, True)

    return data

def get_symbol(num_classes, **kwargs):
    data = mx.sym.var('data')
    data = mx.sym.Convolution(data=data, num_filter=24, 
                              kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max', 
                          stride=(2, 2), pad=(1, 1))
    
    data = make_stage(data, 2)
    
    data = make_stage(data, 3)
    
    data = make_stage(data, 4)
     
    data = mx.sym.Pooling(data=data, kernel=(1, 1), global_pool=True, pool_type='avg')
    
    data = mx.sym.flatten(data=data)
    
    data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)
    
    out = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return out
