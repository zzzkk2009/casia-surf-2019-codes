
## 1、代码和数据集存放
    1.1 获取数据集，即CASIA-SURF文件夹，里面包含phase1和phase2两个文件夹，进入phase1，解压train.zip和valid.zip文件，得到Training文件夹，train_list.txt文件、Val文件夹、val_public_list.txt文件；
    1.2 将本项目整个文件夹casia-surf-2019-codes拷贝到CASIA-SURF目录里面，保证和phase1同目录；

## 2、环境安装(ubuntu 16.04):
    2.1 打开命令终端，cd到casia-surf-2019-codes目录，执行： 
       ```Shell
      pip install -r requirements.txt
      ```

## 3、数据集预处理：
    3.1 打开命令终端,cd到casia-surf-2019-codes目录，依次执行：
      3.1.1 ```Shell python data_preprocess.py ```  （生成val数据集的list文件）
      3.1.2 ```Shell python data_preprocess.py --train ``` （生成train数据集的list）
      3.1.3 ```Shell cd data ```
      3.1.4 ```Shell python im2rec.py train_depth_all_112_29266.lst ../../phase1 ``` （生成train数据集的.rec文件，用于mxnet训练）
      3.1.5 ```Shell python im2rec.py val_depth_all_112_9608.lst ../../phase1 ``` （生成val数据集的.rec文件，用于mxnet验证）

## 4、模型训练
    4.1 打开命令终端，cd到casia-surf-2019-codes目录，执行： 
     ```Shell
    python train_depth_shufflenet_v2.py
    ```
	注：1) 本人使用的是包含4块GPU的服务器，配置为：11G Memory, GeForce GTX 1080，
	    服务器配置：CPU：Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz 12核，64G内存，1TB固态硬盘；
	    2) 训练过程使用了1,2,3块GPU进行训练，batch_size=2048；
	
## 5、测试
    5.1 打开命令终端，cd到casia-surf-2019-codes目录，执行： 
    ```Shell
    python commit.py ../phase1/val_public_list.txt --load-epoch 554
    ```
	注：程序执行完后，会在当前目录下生成 commit_depth_%Y-%m-%d.txt 格式的结果文件
	    commit.py 第一个参数为待提交成绩的list文件，第二个参数load-epoch表示：加载第load-epoch数的模型进行预测。

