# -*- coding:utf-8 -*-

import argparse,logging,os
import mxnet as mx
from symbols.densenet import DenseNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

def multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90, 95, 115, 120], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def main():
    if args.data_type == "imagenet":
        args.num_classes = 1000
        if args.depth   == 121:
            units = [6, 12, 24, 16]
        elif args.depth == 169:
            units = [6, 12, 32, 32]
        elif args.depth == 201:
            units = [6, 12, 48, 32]
        elif args.depth == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = DenseNet(units=units, num_stage=4, growth_rate=48 if args.depth == 161 else args.growth_rate, num_class=args.num_classes, 
                            data_type="imagenet", reduction=args.reduction, drop_out=args.drop_out, bottle_neck=True,
                            bn_mom=args.bn_mom, workspace=args.workspace)
    elif args.data_type == "vggface":
        args.num_classes = 2613
        if args.depth   == 121:
            units = [6, 12, 24, 16]
        elif args.depth == 169:
            units = [6, 12, 32, 32]
        elif args.depth == 201:
            units = [6, 12, 48, 32]
        elif args.depth == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = DenseNet(units=units, num_stage=4, growth_rate=48 if args.depth == 161 else args.growth_rate, num_class=args.num_classes, 
                            data_type="vggface", reduction=args.reduction, drop_out=args.drop_out, bottle_neck=True,
                            bn_mom=args.bn_mom, workspace=args.workspace)
    elif args.data_type == "msface":
        args.num_classes = 79051
        if args.depth   == 121:
            units = [6, 12, 24, 16]
        elif args.depth == 169:
            units = [6, 12, 32, 32]
        elif args.depth == 201:
            units = [6, 12, 48, 32]
        elif args.depth == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = DenseNet(units=units, num_stage=4, growth_rate=48 if args.depth == 161 else args.growth_rate, num_class=args.num_classes, 
                            data_type="msface", reduction=args.reduction, drop_out=args.drop_out, bottle_neck=True,
                            bn_mom=args.bn_mom, workspace=args.workspace)
	
    elif args.data_type == "casia-surf":
        args.num_classes = 2
        if args.depth   == 121:
            units = [6, 12, 24, 16]
        elif args.depth == 169:
            units = [6, 12, 32, 32]
        elif args.depth == 201:
            units = [6, 12, 48, 32]
        elif args.depth == 161:
            units = [6, 12, 36, 24]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = DenseNet(units=units, num_stage=4, growth_rate=48 if args.depth == 161 else args.growth_rate, num_class=args.num_classes, 
                            data_type="casia-surf", reduction=args.reduction, drop_out=args.drop_out, bottle_neck=True,
                            bn_mom=args.bn_mom, workspace=args.workspace)
    else:
        raise ValueError("do not support {} yet".format(args.data_type))
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/densenet-{}-{}-{}".format(args.data_type, args.depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)
    
    # import pdb
    # pdb.set_trace()

    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train_depth_all_112_29266.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 112, 112),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet and vggface, 384 with msface, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.667,  # 256.0/480.0=0.533, 256.0/384.0=0.667
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "val_depth_all_112_9608.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 112, 112),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
	
    model = mx.model.FeedForward(
        ctx                 = devs,
        symbol              = symbol,
        arg_params          = arg_params,
        aux_params          = aux_params,
        num_epoch           = 200 if args.data_type == "cifar10" else 125,
        begin_epoch         = begin_epoch,
        learning_rate       = args.lr,
        momentum            = args.mom,
        wd                  = args.wd,
        optimizer           = 'nag',
        # optimizer          = 'sgd',
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=[220, 260, 280], factor=0.1)
                             if args.data_type=='cifar10' else
                             multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90, 95, 115, 120], factor=0.1),
        )
	
    # import pdb
    # pdb.set_trace()

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = ['acc', 'f1'],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint)
    #logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training DenseNet-BC")
    parser.add_argument('--gpus', type=str, default='1,2,3', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='casia-surf', help='the dataset type')
    parser.add_argument('--list-dir', type=str, default='./', help='the directory which contain the training list file')
    parser.add_argument('--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
    parser.add_argument('--growth-rate', type=int, default=32, help='the growth rate of DenseNet')
    parser.add_argument('--drop-out', type=float, default=0.2, help='the probability of an element to be zeroed')
    parser.add_argument('--reduction', type=float, default=0.5, help='the compression ratio for TransitionBlock')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=201, help='the depth of resnet')
    parser.add_argument('--num-classes', type=int, default=2, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=29266, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    if not os.path.exists("./log"):
        os.mkdir("./log")
    hdlr = logging.FileHandler('./log/log-densenet-{}-{}.log'.format(args.data_type, args.depth))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logging.info(args)
    main()