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

def cmp(var1, var2):
    tmp = 0
    if var1 >= 0.5:
        tmp = 1.
    else:
        tmp = 0.
    if (tmp == var2):
        return 1.
    else:
        return 0.

def F1(var1, var2):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i,j in zip(var1, var2):
		if(i >= 0.5):
			if(j == 1.):
				TP += 1
			else:
				FP += 1
		else:
			if(j == 0.0):
				TN += 1
			else:
				FN += 1
	return (TP, FP, TN, FN)

with tf.Session() as sess:

    embeddings = tf.Variable(embed, name='embeddings',trainable=False)


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
    with tf.variable_scope("cnn_sv", reuse=True):
        sentence_to_vec2 = cnn_sv(input_layer2,FLAGS) #句子的向量表示


    output = tf.exp(-tf.reduce_sum(tf.abs(sentence_to_vec1 - sentence_to_vec2), axis=1, keep_dims=False))
    output = tf.reshape(output,[-1]) #句向量相乘得到输出

    loss = tf.losses.log_loss(similarity, output)

    train_step = tf.train.AdamOptimizer(1e-1).minimize(loss)

    init = tf.global_variables_initializer()
    init.run()

    train_dataset = ReadDataset(FLAGS.train_dir)
    test_dataset = ReadDataset(FLAGS.test_dir)
    start_time = time.time()
    sp = start_time

    # summarywriter = tf.summary.FileWriter('./logs', sess.graph)
    saver = tf.train.Saver()

    saver.restore(sess, './ckpt/cnn.ckpt')
    best = 0
    bstep = 0



    for i in range(1000):
        x, y, z= train_dataset.next_batch(FLAGS.batch_size)
        train_step.run({s1: x, s2: y, similarity: z})

        if ((i+1)%10 == 0):
            loss_value= sess.run(loss, feed_dict={s1: x, s2: y, similarity: z})
            point = time.time() - sp
            test_dataset = ReadDataset(FLAGS.test_dir)
            prediction = []
            label = []
            for j in range(27):  # 69
                x1, y1, z1 = test_dataset.next_batch(FLAGS.batch_size)
                o = sess.run(output, feed_dict={s1: x1, s2: y1, similarity: z1})
                prediction.extend(o)
                label.extend(z1)

                prediction = prediction[:1725]
                label = label[:1725]
            result = map(cmp, prediction, label)
            accuary = sum(result) / len(result)

            TP, FP, TN, FN = F1(prediction, label)
            P = TP/float(TP + FP)
            R = TP/float(TP + FN)
            f1 = 2.*P*R/(P + R)
            if (i + 1 > 0):
                if (accuary > best):
                    best = accuary
                    bstep = i + 1
                    saver.save(sess, './ckpt/cnn.ckpt')

            print("step %d, loss: %f, used time: %f(sec), total accuary:%f" % (i + 1, loss_value, point, accuary))
            print("TP: %d, FP: %d, TN: %d, FN: %d, accuary:%f, F1:%f"%(TP, FP, TN, FN, (TP + TN)/float(TP + TN + FP + FN), f1))

            sp = time.time()

    # saver.save(sess,'./ckpt/cnn.ckpt')
    duration = time.time() - start_time
    print("total cost time:%f(sec)"%duration)
    print("best accuary:%f, step:%d"%(best, bstep))

    #accuary:0.740870, F1:0.820265
    #accuary:0.746087