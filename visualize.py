#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:35:44 2017

@author: yufeng
"""

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib import cm
import os
import cv2

caffe_root='I:/DeepLearn/models/'
model_file='I:/DeepLearn/models/bvlc_reference_caffenet.caffemodel'
#model_file=caffe_root+'/models/bvlc_reference_caffenet/deploy.prototxt'
model_def_file='I:/DeepLearn/models/bvlc_reference_caffenet_deploy.prototxt'
local_root=os.getcwd()#os.path.dirname(os.path.realpath(__file__))
print(local_root)
image_dim=227
grid_size=128

def classify(net, image_file):
    imagenet_labels_filename = 'synset_words.txt'
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')  
    #对目标路径中的图像，遍历并分类
    input_image = np.zeros((len(image_file), 256, 256,3))
    for i in range(len(image_file)):
        input_image[i:,:,:]=cv2.resize(cv2.imread(image_file[i]), (256,256), interpolation=cv2.INTER_CUBIC)
        #input_image[i,:,:,:]=caffe.io.load_image(image_file[i])
    prediction = net.predict(input_image)
    print('prediction shape ')
    print(prediction.shape) #(len(image_file), 1000)
    for i in range(prediction.shape[0]):
        top_1=prediction[i].argmax()
        print 'predicted class:',top_1
        # 输出概率最大的前5个预测结果
        top_k = net.blobs['prob'].data[i].flatten().argsort()[-1:-6:-1]
        print labels[top_k]

def forward(net, image_file):
    input_image = np.zeros((len(image_file), 3, image_dim, image_dim))
    for i in range(len(image_file)):
        input_image[i,:,:,:,]=cv2.resize(cv2.imread(image_file[i]), (image_dim,image_dim), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
    output_data=net.forward_all(**{net.inputs[0]:input_image})
    output_data=output_data[net.outputs[0]]
    print('output shape:')
    print(output_data.shape)
        

"""
对于参数blob，可以将其可视化为rgb图像。其维度变化过程为(从data到可视化结果)
N,c,h,w-->N,h,w,c-->n,n,h,w,c-->n,h,n,w,c-->nh,nw,c
特征图blob的维度是(N,h,w)，只能可视化为灰度图像。其维度变化过程为(从data到可视化结果)
N,h,w-->n,n,h,w-->n,h,n,w-->nh,nw

"""
def vis_square(data, padsize=1, padval=0, use_cv2=1, save_path=None):
    dim=data.ndim
    data=data.astype(np.float32)
    if dim==4:
        _data=np.zeros((data.shape[0], data.shape[1], grid_size, grid_size), dtype=np.float32)
    elif dim==3:
        _data=np.zeros((data.shape[0], grid_size, grid_size), dtype=np.float32)
        
    if dim==4 and use_cv2:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                _data[i,j,:,:]=cv2.resize(data[i,j,:,:], (grid_size,grid_size), interpolation=cv2.INTER_CUBIC)

    elif dim==3 and use_cv2:
        for i in range(data.shape[0]):
            _data[i,:,:]=cv2.resize(data[i,:,:], (grid_size,grid_size), interpolation=cv2.INTER_CUBIC)
    if use_cv2:
        data=_data
    
    if dim==4:
        data=data.transpose(0,2,3,1)
    
    """normalize to (0,1) float32"""
    print('max value: %f and min value: %f'%(data.max(), data.min()))
    data_min=data.min()
    data -= data_min
    data_max_sub=data.max()
    data /= data.max()
    
    #让合成图为方
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #合并卷积图到一个图像中
    print(str((n,n)+data.shape[1:]))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + (4,)*(data.ndim-3))
    print(data.shape)
    if dim==4:
        new_shape=(n*data.shape[1],n*data.shape[3],data.shape[4])
    elif dim==3:
        new_shape=(n*data.shape[1], n*data.shape[3])
    data = data.reshape(new_shape)
    print data.shape
    if not use_cv2:
        plt.imshow(data, cmap=cm.get_cmap('jet'))
        plb.show()
    if use_cv2:
        if save_path is not None:
            data*=data_max_sub
            data+=data_min
            data=data.astype(np.uint8)
            cv2.imwrite(save_path, data)
        """
        opencv显示图像要么是float32要么是uint8(3).转化为int32后保存到文件系统，
        实际上和8uc3是一样的，但是不能imshow
        """
        cv2.imshow('data', data)
        cv2.waitKey()
        cv2.destroyAllWindows()

def vis_surface(data, padsize=1, padval=0):
    surface_data=np.zeros((data.shape[0],1,data.shape[2], data.shape[3]))
    surface_data[:,0,:,:]=data[:,0,:,:]
    print(surface_data.shape)
    vis_square(surface_data)


def vis_histogram(data):
    print('fc data shape: '+str(data.shape))
    #plt.subplot(211)
    plt.figure()
    plt.plot(data.flat)

def show_blob_structure(net):
    n=0
    print('blob structure')
    for blob_name, blob in net.blobs.iteritems():
        print('blob '+str(n)+' name: '+blob_name+', shape: '+str(blob.data.shape))
        n+=1

def show_layer_structure(net):
    n=0
    print('layer structure')
    layer_names=list(net._layer_names)
    for layer_name in layer_names:
        print('layer '+str(n)+' name: '+layer_name)
        n+=1
    
def show_param_structure(net):
    n=0
    print('param structure')
    for layer_name, param in net.params.iteritems():
        print('layer '+str(n)+' name: '+layer_name+', shape: '+str(param[0].data.shape))
        n+=1
    
if __name__=='__main__':
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_def_file, model_file,
                   mean=np.load(caffe_root + 'ilsvrc_2012_mean.npy').mean(1).mean(1),
                   channel_swap=(0,1,2),
                   raw_scale=255,
                   image_dims=(256, 256))
    _net=caffe.Net(model_def_file, model_file, caffe.TEST)
    image_path=[os.path.join(local_root,'face.jpg')]#, os.path.join(local_root,'cat.jpg')]
    show_layer_structure(net)
    show_blob_structure(net)
    show_param_structure(net)
    forward(_net, image_path)
    #classify(net, image_path)
	#params[0]=conv_param
	#params[1]=bias_param
    conv=_net.params['conv1'][0].data
    print('conv1 shape: ')
    #print(net.params['conv1_1'])
    print('origin shape: '+str(conv.shape))
    fm=_net.blobs['conv1_top'].data[0,:36]
    print('pool shape: '+str(fm.shape))
    fc=_net.blobs['fc7'].data[0]
    #fc_param=net.params['fc7'][0].data
    vis_square(conv)
    vis_square(fm)
    #vis_surface(conv)
    #vis_histogram(fc)
    