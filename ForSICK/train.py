# -*- coding: utf-8 -*-
'''这是CNN_SV模型的训练文件'''
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from read import ReadDataset
import read
from model import*
import time

FLAGS = Flags()#参数设置
embed = read.embeddings
word_len = len(embed[0])#词向量的维度
FLAGS.len_of_word_vector = word_len

s1 = tf.placeholder(tf.int32,[None,None,None])#承接句子,句子的表示比较特殊方便后面lookup和得到输入层
s2 = tf.placeholder(tf.int32,[None,None,None])
similarity = tf.placeholder(tf.float32, [None])

with tf.Session() as sess:
    embeddings = tf.Variable(embed, name='embeddings')

    v1 = tf.Variable(0.4,name='v1')
    v2 = tf.Variable(0.6,name='v2')

    w1 = tf.exp(v1)/(tf.exp(v2)+tf.exp(v1))
    w2 = 1. - w1#对词向量进行加权的权值

    s11 = w1*tf.nn.embedding_lookup(embeddings, s1)
    s12 = w2*tf.transpose(s11, [0, 2, 1, 3])
    input_layer1 = tf.add(s11, s12)#输入层

    s21 = w1*tf.nn.embedding_lookup(embeddings, s2)
    s22 = w2*tf.transpose(s21, [0, 2, 1, 3])
    input_layer2 = tf.add(s21, s22)

    with tf.variable_scope("cnn_sv") as scope:
        sentence_to_vec1 = cnn_sv(input_layer1,FLAGS) #句子的向量表示
    with tf.variable_scope(scope, reuse=True):
        sentence_to_vec2 = cnn_sv(input_layer2,FLAGS) #句子的向量表示

    cosine_distance = tf.matmul(sentence_to_vec1, sentence_to_vec2,transpose_a=True) #3x3
    cosine_distance = tf.reshape(cosine_distance,[-1, FLAGS.max_k*FLAGS.max_k])
    k_cosine_distance, _ = tf.nn.top_k(cosine_distance, k=FLAGS.max_k)
    output = 5*tf.reduce_mean(k_cosine_distance,1)

    loss = tf.reduce_mean(tf.square(similarity - output))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    _, train_pearson = tf.contrib.metrics.streaming_pearson_correlation(output, similarity)
    _, test_pearson = tf.contrib.metrics.streaming_pearson_correlation(output, similarity) #adding

    train_pearson_summary = tf.summary.scalar("train_pearson", train_pearson)
    test_pearson_summary = tf.summary.scalar("test_pearson", test_pearson)

    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    init.run()

    train_dataset = ReadDataset(FLAGS.train_dir)
    test_dataset = ReadDataset(FLAGS.test_dir)
    start_time = time.time()
    sp = start_time

    # summarywriter = tf.summary.FileWriter('./logs', sess.graph)
    # saver = tf.train.Saver()
    # saver.restore(sess, './ckpt/cnn.ckpt')

    for i in range(3600):
        x, y, z= train_dataset.next_batch(FLAGS.batch_size)
        if ((i+1)%90 == 0):
            l, p, train_summary = sess.run([loss, train_pearson, train_pearson_summary],feed_dict={s1: x, s2: y, similarity: z}) # changed: add pearson
            # summarywriter.add_summary(train_summary, i)
            test_s1, test_s2, test_label = test_dataset.next_batch(FLAGS.batch_size)
            test_l, test_p, test_summary = sess.run([loss, test_pearson, test_pearson_summary],feed_dict={s1: test_s1, s2: test_s2, similarity: test_label}) 
            # summarywriter.add_summary(test_summary, i)
            point = time.time() - sp
            print("step %d, loss: %f, test loss: %f, used time: %f(sec), pearson:%f, test pearson: %f"%(i+1, l, test_l, point, p, test_p)) 
            sp = time.time()
        train_step.run({s1: x, s2: y, similarity: z})
    saver = tf.train.Saver()
    saver.save(sess,'./ckpt/cnn.ckpt')
    duration = time.time() - start_time
    print("total cost time:%f(sec)"%duration)