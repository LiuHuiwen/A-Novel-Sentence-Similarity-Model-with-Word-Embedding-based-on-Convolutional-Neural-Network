# -*- coding: utf-8 -*-
'''这是CNN_SV模型用来读取数据集并对数据集进行预处理的文件'''
import numpy as np
import pickle
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

w2v_dir = '../glove.6B.50d.txt'

dictionary = {}
embeddings = []
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '``', '\'\'']
with open(w2v_dir,'rb') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    words = words[:10000]
    dictionary  = {word:(index) for index, word in enumerate(words)}
    del words

with open(w2v_dir,'rb') as f:
    vectors = [[float(y) for y in x.rstrip().split(' ')[1:]] for x in f.readlines()]
    vectors = vectors[:10000]
    embeddings = embeddings + vectors
    del vectors

class ReadDataset:

    def __init__(self, filename):

        f = open(filename, 'rb')
        self.dataset = f.readlines() #read dataset to class ReadDataset
        f.close()
        self.dataset = self.dataset[1:]
        shuffle(self.dataset)

        self.data_index = 0
        self.lines = len(self.dataset) #9840

    def next_batch(self, batch):
        x = [] #sentence one
        y = [] #sentence two 
        z = [] #label
        vector_x = [] #Temporary variable
        vector_y = [] #Temporary variable
        max_len_x = 0
        max_len_y = 0
        for i in range(batch):
            data = self.dataset[self.data_index].split('\t')
            s1 = self.sentence_to_index(data[1]) #return a list of index
            s2 = self.sentence_to_index(data[2])
            l = float(data[3])
            vector_x.append(s1)
            vector_y.append(s2)
            z.append(l)
            max_len_x = max(max_len_x, len(s1))
            max_len_y = max(max_len_y, len(s2))
            self.data_index = (self.data_index + 1)%self.lines
        for j in vector_x:
            x.append(self.vector_to_matrix(j,max_len_x))
        del(vector_x)
        for k in vector_y:
            y.append(self.vector_to_matrix(k,max_len_y))
        del(vector_y)
        return x,y,z # stence label

    def sentence_to_index(self,stc_str):#stc:sentence, str: string (dtype)
        global dictionary # key: word, value: index in embedding
        index_lst = [] #return, list of index represent sentence
        # padding = dictionary['UNK']
        word_lst = stc_str.split()
        for word in word_lst:
            if(dictionary.has_key(word.lower())):
                index_lst.append(dictionary[word.lower()])
        return index_lst

    def vector_to_matrix(self,vector,max_len):
        l = len(vector)
        vector.extend([0 for i in range(max_len-l)])
        maxtrix = [vector for j in range(l)]
        maxtrix.extend([[0 for k in range(max_len)] for m in range(max_len-l)])
        return maxtrix