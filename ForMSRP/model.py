# -*- coding: utf-8 -*-
'''这是CNN_SV的模型代码，由于输入层的部分处理使用tensorflow不太方便,放在了read.py中'''
import tensorflow as tf
import numpy as np
import pickle

class Flags:#存储模型所需的所有参数
    def __init__(self):
        self.train_dir = './dataset/msr_paraphrase_train.txt'
        self.test_dir = './dataset/msr_paraphrase_test.txt'
        self.len_of_word_vector = 100
        self.len_of_sentence_vector =100 #used
        self.filter_para = 3
        self.batch_size = 64
        self.max_k = 3

def conv2d(x, shape): #x为输入
    W = tf.get_variable(name='weight', shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b = tf.get_variable(name='bias', shape=shape[-1:], dtype=tf.float32, initializer=tf.constant_initializer(0))
    conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    return conv

def cnn_sv(x,FLAGS):
    conv1 = conv2d(x, [FLAGS.filter_para, FLAGS.filter_para, FLAGS.len_of_word_vector, FLAGS.len_of_sentence_vector])

    pool2_flat = tf.reshape(conv1, [FLAGS.batch_size,-1, FLAGS.len_of_sentence_vector])#将二维图变成一维的向量组,三维
    pool2_flat = tf.transpose(pool2_flat,[0,2,1])#变形，方便max-k池化
    values, indices = tf.nn.top_k(pool2_flat, k=FLAGS.max_k)

    sv = tf.reduce_mean(values, axis=2)#对提取出的向量取平均,此时最后一维消失，得到非正则化的sentence vector,二维

    return sv