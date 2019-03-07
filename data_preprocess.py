# -*- coding:utf-8 -*-

import os
import random
import cv2
import argparse

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

class CasiaSurf(object):

    def __init__(self):
        self.color_list = []
        self.depth_list = []
        self.ir_list = []

    def preprocess_val_list(self):
        positive_num = 0
        negative_num = 0
        # mkdir_if_not_exist(['data'])
        # color_wf =open('data/val_color_all_112_{}.lst'.format(9608), 'w')
        depth_wf =open('data/val_depth_all_112_{}.lst'.format(9608), 'w')
        # ir_wf =open('data/val_ir_all_112_{}.lst'.format(9608), 'w')
        with open('val_public_list_with_label.txt', 'r') as f:
            i = 0
            for line in f.readlines():
                line = line.strip() # 去掉每行头尾空白
                line_lst = line.split() # 按空白符分割

                # color_path = line_lst[0]
                # resize_color_path = 'Val-112/' + color_path.split('/', 1)[1]
                # colr_img = cv2.imread('../phase1/' + color_path)
                # colr_img_resized = cv2.resize(colr_img,(112,112))
                # color_path_lst = ['..', 'phase1']
                # color_path_lst.extend(resize_color_path.split('/')[0:-1])
                # mkdir_if_not_exist(color_path_lst)
                # cv2.imwrite('../phase1/' + resize_color_path, colr_img_resized)

                depth_path = line_lst[1]
                resize_depth_path = 'Val-112/' + depth_path.split('/', 1)[1]
                # depth_img = cv2.imread('../phase1/' + depth_path)
                # depth_img_resized = cv2.resize(depth_img, (112,112))
                # depth_path_lst = ['..', 'phase1']
                # depth_path_lst.extend(resize_depth_path.split('/')[0:-1])
                # mkdir_if_not_exist(depth_path_lst)
                # cv2.imwrite('../phase1/' + resize_depth_path, depth_img_resized)

                # ir_path = line_lst[2]
                # resize_ir_path = 'Val-112/' + ir_path.split('/', 1)[1]
                # ir_img = cv2.imread('../phase1/' + ir_path)
                # ir_img_resized = cv2.resize(ir_img, (112,112))
                # ir_path_lst = ['..', 'phase1']
                # ir_path_lst.extend(resize_ir_path.split('/')[0:-1])
                # mkdir_if_not_exist(ir_path_lst)
                # cv2.imwrite('../phase1/' + resize_ir_path, ir_img_resized)

                label = line_lst[3]

                if float(label) <= 0.5:
                    negative_num += 1
                else:
                    positive_num += 1

                # color_path_with_label = str(i) + '\t' + label + '\t' + resize_color_path + '\n'
                # i += 1
                depth_path_with_label = str(i) + '\t' + label + '\t' + resize_depth_path + '\n'
                i += 1
                # ir_path_with_label = str(i) + '\t' + label + '\t' + resize_ir_path + '\n'
                # i += 1

                # color_wf.write(color_path_with_label)
                depth_wf.write(depth_path_with_label)
                # ir_wf.write(ir_path_with_label)

                # print('process val line->%d' %(i // 3))
                print('process val line->%d' %(i))
        
        # color_wf.close()
        depth_wf.close()
        # ir_wf.close()

        print('preprocess val list success!')

    def preprocess_train_list(self):
        positive_num = 0
        negative_num = 0

        with open('../phase1/train_list.txt', 'r') as f:
            i = 0
            for line in f.readlines():
                line = line.strip() # 去掉每行头尾空白
                line_lst = line.split() # 按空白符分割

                # color_path = line_lst[0]
                # resize_color_path = 'Training-112/' + color_path.split('/', 1)[1]
                # colr_img = cv2.imread('../phase1/' + color_path)
                # colr_img_resized = cv2.resize(colr_img,(112,112))
                # color_path_lst = ['..', 'phase1']
                # color_path_lst.extend(resize_color_path.split('/')[0:-1])
                # mkdir_if_not_exist(color_path_lst)
                # cv2.imwrite('../phase1/' + resize_color_path, colr_img_resized)

                depth_path = line_lst[1]
                resize_depth_path = 'Training-112/' + depth_path.split('/', 1)[1]
                # depth_img = cv2.imread('../phase1/' + depth_path)
                # depth_img_resized = cv2.resize(depth_img, (112,112))
                # depth_path_lst = ['..', 'phase1']
                # depth_path_lst.extend(resize_depth_path.split('/')[0:-1])
                # mkdir_if_not_exist(depth_path_lst)
                # cv2.imwrite('../phase1/' + resize_depth_path, depth_img_resized)

                # ir_path = line_lst[2]
                # resize_ir_path = 'Training-112/' + ir_path.split('/', 1)[1]
                # ir_img = cv2.imread('../phase1/' + ir_path)
                # ir_img_resized = cv2.resize(ir_img, (112,112))
                # ir_path_lst = ['..', 'phase1']
                # ir_path_lst.extend(resize_ir_path.split('/')[0:-1])
                # mkdir_if_not_exist(ir_path_lst)
                # cv2.imwrite('../phase1/' + resize_ir_path, ir_img_resized)

                label = line_lst[3]

                if 0 == int(label):
                    negative_num += 1
                else:
                    positive_num += 1

                # color_path_with_label = str(i) + '\t' + label + '\t' + resize_color_path
                # i += 1
                depth_path_with_label = str(i) + '\t' + label + '\t' + resize_depth_path
                i += 1
                # ir_path_with_label = str(i) + '\t' + label + '\t' + resize_ir_path
                # i += 1
                
                # self.color_list.append(color_path_with_label)
                self.depth_list.append(depth_path_with_label)
                # self.ir_list.append(ir_path_with_label)

                # print('process train line->%d' %(i // 3))
                print('process train line->%d' %(i))
                
        print('positive_num=%d' %(positive_num))  # 8942
        print('negative_num=%d' %(negative_num))  # 20324

        # random.shuffle(self.color_list)
        random.shuffle(self.depth_list)
        # random.shuffle(self.ir_list)

        # with open('train_color_all_112_{}.lst'.format(len(self.color_list)), 'w') as f:
        #     f.write('\n'.join(self.color_list))
        with open('data/train_depth_all_112_{}.lst'.format(len(self.depth_list)), 'w') as f:
            f.write('\n'.join(self.depth_list))
        # with open('train_ir_all_112_{}.lst'.format(len(self.ir_list)), 'w') as f:
        #     f.write('\n'.join(self.ir_list))

        print('preprocess train list success!')

    def use_train_sublist(self):
        positive_num = 0
        negative_num = 0

        with open('data/train_depth_all_112_29266.lst', 'r') as f:
            i = 0
            for line in f.readlines():
                line = line.strip() # 去掉每行头尾空白
                line_lst = line.split() # 按空白符分割

                label = int(line_lst[1])
                depth_path = line_lst[2]
                
                if 0 == label:
                    if not '_enm_' in depth_path:
                        self.depth_list.append(line)
                        negative_num += 1
                else:
                    positive_num += 1
                    self.depth_list.append(line)

                i += 1
                print('process train line->%d' %(i))
                
        print('positive_num=%d' %(positive_num))  # 8942
        print('negative_num=%d' %(negative_num))  # 6518

        random.shuffle(self.depth_list)

        with open('data/train_depth_noenmfake_112_{}.lst'.format(len(self.depth_list)), 'w') as f:
            f.write('\n'.join(self.depth_list))

        print('preprocess train sublist success!')

    def aug_trainlist(self):

        with open('data/train_depth_all_112_29266.lst', 'r') as f:
            i = 0
            pLst = []
            nLst = []
            allLst = []
            for line in f.readlines():
                line = line.strip() # 去掉每行头尾空白
                line_lst = line.split() # 按空白符分割

                label = int(line_lst[1])
                # depth_path = line_lst[2]
                
                if 0 == label:
                    nLst.append(line)
                else:
                    pLst.append(line)

                i += 1
                print('process aug train line->%d' %(i))
                
        print('positive_num=%d' %(len(pLst)))  # 8942
        print('negative_num=%d' %(len(nLst)))  # 20324

        # augPLst = random.sample(pLst, 8942)
        augPLst = pLst[:] # 拷贝自身
        print('augPLst num=%d' %(len(augPLst)))
        pLst.extend(augPLst)
        print('positive_num=%d' %(len(pLst)))  # 17884
        print('negative_num=%d' %(len(nLst)))  # 20324

        allLst = pLst + nLst
        print('allLst num=%d' %(len(allLst))) # 38208
        random.shuffle(allLst)

        with open('data/train_depth_aug_112_{}.lst'.format(len(allLst)), 'w') as f:
            f.write('\n'.join(allLst))

        print('preprocess aug trainlist success!')

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    # parser.add_argument('--train', action='store_true',
    #     help='generate train/val list file & resize train/val image to 112 size which saved in ../phase1/ dir.')
    parser.add_argument('train', help='generate train/val list file & resize train/val image to 112 size which saved in ../phase1/ dir.')
    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--no-enmfake', action='store_true', default=False,
        help='remove enm fake train image dataset')
    cgroup.add_argument('--aug', action='store_true', default=False,
        help='augment train positive image dataset')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    casiaSurf = CasiaSurf()
    # mkdir_if_not_exist(['data'])
    if args.train == 'train':
        if args.no_enmfake:
            casiaSurf.use_train_sublist()
        elif args.aug:
            casiaSurf.aug_trainlist()
        else:
            casiaSurf.preprocess_train_list()
    else:
        casiaSurf.preprocess_val_list()
    
