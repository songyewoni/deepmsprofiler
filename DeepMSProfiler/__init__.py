import sys

import pandas as pd
import numpy as np
from pyteomics import mzml
import shutil
import os
import time
from sklearn.model_selection import train_test_split


from .mainRun import step1, step2_train,load_jobs_data,step2_pred,step2_feature
from .utils import loadKerasModel
import argparse



def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')

    parser.add_argument('-data', type=str, dest='data_dir', default='../example/data/')
    parser.add_argument('-split', type=str, dest='split_path', default=None)
    parser.add_argument('-random_split', type=bool, dest='random_split', default=False)
    parser.add_argument('-out', type=str, dest='out_dir', default='../jobs')

    parser.add_argument('-arch', type=str, dest='architecture', default='DenseNet121')
    parser.add_argument('-nc',dest='num_classes', default=3)
    parser.add_argument('-pretrain',dest='pretrain_path', default=None)
    parser.add_argument('-lr', type=float, dest='learning_rate', default=1e-4)
    parser.add_argument('-opt', type=str, dest='optimizer', default='adam')
    parser.add_argument('-batch', type=int, dest='batch_size', default=8)
    parser.add_argument('-epoch', type=int, dest='epoch', default=200)
    parser.add_argument('-run', type=int, dest='run_time', default=10)

    parser.add_argument('-models', type=str, dest='models_for_pred', default='use_old')
    parser.add_argument('-mode', type=str, dest='mode', default='ensemble')
    parser.add_argument('-boost', dest='boost', action='store_true', help='boosting mode')
    parser.add_argument('-plot_auc', dest='plot_auc', action='store_true', help='plot auc curve')
    parser.add_argument('-plot_cm', dest='plot_cm', action='store_true', help='plot confusion matrix')

    args = parser.parse_args()
    return args


def run_train(job_dir='./jobs007',**kwargs):
    args = args_setting()

    for key in kwargs:
        setattr(args, key, kwargs[key])

    print('[INFO] args:')
    for key in args.__dict__:
        print(key,':', args.__dict__[key])

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    print('[INFO] Start in %s!' % job_dir)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Step1 : load data
    step1(args, job_dir)

    print('[INFO] Step1 Done!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


    num_classes = args.num_classes
    train_data, train_label, train_samples, test_data, test_label, test_samples = load_jobs_data(job_dir, num_classes=num_classes)

    step2_train(args, job_dir, train_data, train_label, train_samples)
    print('[INFO] Step2 Train Done!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def run_predict(job_dir, reset_data=False,**kwargs):
    args = args_setting()

    for key in kwargs:
        setattr(args, key, kwargs[key])

    print('[INFO] args:')
    for key in args.__dict__:
        print(key, ':', args.__dict__[key])


    print('[INFO] Start in %s!' % job_dir)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Step1 : load data
    if reset_data:
        step1(args, job_dir)

        print('[INFO] Step1 Done!')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


    num_classes = args.num_classes
    train_data, train_label, train_samples, test_data, test_label, test_samples = load_jobs_data(job_dir,
                                                                                                 num_classes=num_classes)
    step2_pred(args, job_dir, test_data, test_label, test_samples)
    print('[INFO] Step2 Pred Done!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def run_feature(job_dir, reset_data=False,**kwargs):
    args = args_setting()

    for key in kwargs:
        setattr(args, key, kwargs[key])

    print('[INFO] args:')
    for key in args.__dict__:
        print(key, ':', args.__dict__[key])

    print('[INFO] Start in %s!' % job_dir)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Step1 : load data
    if reset_data:
        step1(args, job_dir)

        print('[INFO] Step1 Done!')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    num_classes = args.num_classes
    train_data, train_label, train_samples, test_data, test_label, test_samples = load_jobs_data(job_dir,
                                                                                                 num_classes=num_classes)
    step2_feature(args, job_dir, test_data, test_label, test_samples)
    print('[INFO] Step2 Pred Done!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def show_feature(job_dir,mode = 'ensemble',num_classes=3):
    feature_path = os.path.join(job_dir, 'feature_results', '%s_RISE.npy' % mode)

    train_data, train_label, train_samples, test_data, test_label, test_samples = load_jobs_data(job_dir,
                                                                                                 num_classes=num_classes)

    ## linux
    if sys.platform == 'linux':
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. 加载数据
    feature = np.load(feature_path)
    labels = np.array(test_label)
    labels = np.argmax(labels, axis=1)

    print('Labels >>>')
    print(labels)
    print('Shape of Feature: %s' % str(feature.shape))

    unique_labels = np.unique(labels)
    print(unique_labels)

    i = 1
    for label in unique_labels:
        plt.subplot(1, len(unique_labels), i)
        label_idx = np.where(labels == label)
        print(label_idx)
        label_feature = feature[label_idx]
        label_feature_mean = np.mean(label_feature, axis=0)
        label_feature_mean = np.flip(label_feature_mean, axis=0)

        print(label_feature_mean.shape)

        plt.imshow(label_feature_mean, cmap='jet', interpolation='nearest')
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('m/z')
        plt.ylabel('RT')
        i = i + 1
    print(feature.shape)
    plt.savefig(os.path.join(job_dir, 'feature_results', '%s_RISE.png' % mode))
    plt.show()

