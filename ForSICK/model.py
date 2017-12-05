# -*- coding: utf-8 -*-
'''这是CNN_SV的模型代码，由于输入层的部分处理使用tensorflow不太方便,放在了read.py中'''
import tensorflow as tf
import numpy as np
import pickle

class Flags:#存储模型所需的所有参数
    def __init__(self):
        self.train_dir = './dataset/SICK_train.txt' #used
        self.test_dir = './dataset/SICK_test.txt'  # used
        self.embeddings_dir = '/home/liuhuiwen/glove.6B/embeddings.pkl' #used
        self.len_of_word_vector = 100
        self.len_of_sentence_vector = 100 #used
        self.filter_para = 5
        self.batch_size = 50
        self.max_k = 5

def read_embeddings(embeddings_dir):#读取word2vec
    read_data = open(embeddings_dir, 'rb')
    embeddings = pickle.load(read_data)
    read_data.close()
    lenght = len(embeddings[0])
    embeddings[0] = [0. for i in range(lenght)]
    return embeddings

def conv2d(x, shape): #x为输入
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
    conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    # pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#pool在本函数中并不是必需的
    return conv

def cnn_sv(x,FLAGS):
    conv1 = conv2d(x, [FLAGS.filter_para, FLAGS.filter_para, FLAGS.len_of_word_vector, FLAGS.len_of_sentence_vector])

    pool2_flat = tf.reshape(conv1, [FLAGS.batch_size,-1, FLAGS.len_of_sentence_vector])#将二维图变成一维的向量组,三维
    pool2_flat = tf.transpose(pool2_flat,[0,2,1])#变形，方便max-k池化
    values, indices = tf.nn.top_k(pool2_flat, k=FLAGS.max_k)

    norm = tf.sqrt(tf.reduce_sum(tf.square(values),1,keep_dims=True))
    sv = values / norm #sv contain 3 dim, and one sv have 3 vector
    return sv