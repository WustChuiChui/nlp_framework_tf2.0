import logging

import numpy as np
from common.const import *

def encode_query(data_list, vocab, length=30, padding=PAD):
    res_list = []
    for item in data_list:
        query = item[0]
        word_id_list = []
        for word in query:
            if len(word_id_list) >= length: break
            if word in vocab:
                word_id_list.append(vocab[word])
            else:
                word_id_list.append(vocab[UNK])
        for idx in range(len(query), length):
            word_id_list.append(vocab[padding])
        res_list.append(word_id_list)
    return np.array(res_list)

def encode_query_v2(data_list, vocab, length=30, padding="[PAD]"):
    res_list = []
    for query in data_list[0]:
        word_id_list = []
        for word in query:
            if len(word_id_list) >= length: break
            if word in vocab:
                word_id_list.append(vocab[word])
            else:
                word_id_list.append(vocab["[UNK]"])
        
        for idx in range(len(query), length):
            word_id_list.append(vocab[padding])
        res_list.append(word_id_list)
    return np.array(res_list)

def encodeLabel(data_list, label_id_dic, label_colunm=1):
    res_list = []
    for item in data_list:
        res_list.append(label_id_dic[item[label_colunm]])
    return np.array(res_list)

def encodeTags(data_list, tag_id_dic, length=30, padding="O", label_colunm=1):
    res_list = []
    for pair in data_list:
        tag_id_list = []
        for tag in pair[label_colunm]:
            if len(tag_id_list) >= length: break
            if tag in tag_id_dic:
                tag_id_list.append(tag_id_dic[tag])
            else:
                logging.error("Inval tag in token %s" % (tag))
                exit(-1)
        for idx in range(len(tag_id_list), length):
            tag_id_list.append(tag_id_dic[padding])
        res_list.append(tag_id_list)

    return np.array(res_list)

def fill_feed_batch_data(data, batch_size):
    X, Y = data[0], data[1]
    for idx in range(X.shape[0] // batch_size):
        x_batch = X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = Y[batch_size * idx: batch_size * (idx + 1)]
        yield np.array(x_batch), np.array(y_batch)

def fill_feed_batch_data_v2(data, batch_size):
    data_cnt = data[0].shape[0]

    for idx in range(data_cnt // batch_size):
        batch_data = []
        for i in range(len(data)):
            item = data[i][batch_size * idx: batch_size * (idx + 1)]
            batch_data.append(np.array(item))
        yield batch_data
