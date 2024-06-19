import sys
import os

import numpy as np

sys.path.append("../")
from utils.tf_utils import *
from common.const import *
from config.config_parser import ConfigParser
import params_flow as pf
import bert
import json
from abc import abstractmethod

class DataPreprocess():
    def __init__(self):
        self.base_path = "./"
        self.intent_id_dic_file = "intent_id_json"
        self.id_intent_dic_file = "id_intent_json"
        self.tag_id_dic_file = "tag_id_json"
        self.id_tag_dic_file = "id_tag_json"

    def make_output_dir(self, config):
        self.base_path = config.trainer_info.results_dir
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.base_path = self.base_path + "/" + config.task_info.name
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.base_path = self.base_path + "/" + config.task_info.task
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.base_path = self.base_path + "/" + config.task_info.version
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

    def load_json_corpus(self, file_name):
        res = []
        with open(file_name, "r") as file:
            for line in file.readlines():
                res.append(json.loads(line.strip()))
        return res

    def load_albert_vocab(self, config):
        albert_zh_vocab_url = "https://raw.githubusercontent.com/brightmart/albert_zh/master/albert_config/vocab.txt"
        vocab_file = pf.utils.fetch_url(albert_zh_vocab_url, "./bert/models/")
        tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file)
        vocab = tokenizer.vocab

        vocab_file = os.path.join(self.base_path, "vocab.json")
        json.dump(vocab, open(vocab_file, 'w'), ensure_ascii=False)
        print("Vocabulary size: {:d}".format(len(vocab)))

        return vocab, len(vocab)

    def generateLabelMap(self, data_list, label_colunm=1):
        label_id_dic = {}
        id_label_dic = {}

        for data_pair in data_list:
            label = data_pair[label_colunm]
            if label not in label_id_dic:
                label_id_dic[label] = len(label_id_dic)
                id_label_dic[len(id_label_dic)] = label

        return label_id_dic, id_label_dic

    def generateTagsMap(self, data_list, label_colunm=1):
        tag_id_dic = {}
        id_tag_dic = {}
        for pair in data_list:
            for tag in pair[label_colunm]:
                if tag in tag_id_dic:  continue
                tag_id_dic[tag] = len(tag_id_dic)
                id_tag_dic[len(id_tag_dic)] = tag
        return tag_id_dic, id_tag_dic

    @abstractmethod
    def load_corpus(self, config):
        pass

    @abstractmethod
    def __call__(self, config):
        pass

class ClassifyDataPreprocess(DataPreprocess):
    def __init__(self):
        super(ClassifyDataPreprocess, self).__init__()

    def load_corpus(self, config):
        train_data_file = config.corpus_info.train_data_file
        print("Loading train data from {} ...".format(train_data_file))
        train_json_data = self.load_json_corpus(train_data_file)
        train_data = [(item["query"], item["intent"]) for item in train_json_data]

        dev_data_file = config.corpus_info.dev_data_file
        print("Loading dev data from {} ...".format(dev_data_file))
        dev_json_data = self.load_json_corpus(dev_data_file)
        dev_data = [(item["query"], item["intent"]) for item in dev_json_data]

        test_data_file = config.corpus_info.test_data_file
        print("Loading test data from {} ...".format(test_data_file))
        test_json_data = self.load_json_corpus(test_data_file)
        test_data = [(item["query"], item["intent"]) for item in test_json_data]

        return train_data, dev_data, test_data

    def __call__(self, config):
        self.make_output_dir(config)
        train_data, dev_data, test_data = self.load_corpus(config)

        vocab, vocab_size = self.load_albert_vocab(config)
        max_seq_len = config.encoder_info.max_seq_len

        train_data_x = encode_query(train_data, vocab, length=max_seq_len)
        dev_data_x = encode_query(dev_data, vocab, length=max_seq_len)
        test_data_x = encode_query(test_data, vocab, length=max_seq_len)

        intent_id_dic, id_intent_dic = self.generateLabelMap(train_data)

        intent_id_dic_file = os.path.join(self.base_path, self.intent_id_dic_file)
        id_intent_dic_file = os.path.join(self.base_path, self.id_intent_dic_file)
        json.dump(intent_id_dic, open(intent_id_dic_file, "w"), ensure_ascii=False)
        json.dump(id_intent_dic, open(id_intent_dic_file, "w"), ensure_ascii=False)

        train_data_y = encodeLabel(train_data, intent_id_dic)
        dev_data_y = encodeLabel(dev_data, intent_id_dic)
        test_data_y = encodeLabel(test_data, intent_id_dic)

        return (train_data_x, train_data_y), (dev_data_x, dev_data_y), (test_data_x, test_data_y), vocab_size

class NerDataPreprocess(DataPreprocess):
    def __init__(self):
        super(NerDataPreprocess, self).__init__()

    def load_corpus(self, config):
        train_data_file = config.corpus_info.train_data_file
        print("Loading train data from {} ...".format(train_data_file))
        train_json_data = self.load_json_corpus(train_data_file)
        train_data = [(item["query"], item["tags"]) for item in train_json_data]

        dev_data_file = config.corpus_info.dev_data_file
        print("Loading dev data from {} ...".format(dev_data_file))
        dev_json_data = self.load_json_corpus(dev_data_file)
        dev_data = [(item["query"], item["tags"]) for item in dev_json_data]

        test_data_file = config.corpus_info.test_data_file
        print("Loading test data from {} ...".format(test_data_file))
        test_json_data = self.load_json_corpus(test_data_file)
        test_data = [(item["query"], item["tags"]) for item in test_json_data]

        return train_data, dev_data, test_data

    def __call__(self, config):
        self.make_output_dir(config)
        train_data, dev_data, test_data = self.load_corpus(config)

        vocab, vocab_size = self.load_albert_vocab(config)
        max_seq_len = config.encoder_info.max_seq_len

        train_data_x = encode_query(train_data, vocab, length=max_seq_len)
        dev_data_x = encode_query(dev_data, vocab, length=max_seq_len)
        test_data_x = encode_query(test_data, vocab, length=max_seq_len)

        tag_id_dic, id_tag_dic = self.generateTagsMap(train_data)
        tag_id_dic_file = os.path.join(self.base_path, self.tag_id_dic_file)
        id_tag_dic_file = os.path.join(self.base_path, self.id_tag_dic_file)
        json.dump(tag_id_dic, open(tag_id_dic_file, "w"), ensure_ascii=False)
        json.dump(id_tag_dic, open(id_tag_dic_file, "w"), ensure_ascii=False)

        train_data_y = encodeTags(train_data, tag_id_dic, length=max_seq_len)
        dev_data_y = encodeTags(dev_data, tag_id_dic, length=max_seq_len)
        test_data_y = encodeTags(test_data, tag_id_dic, length=max_seq_len)

        return (train_data_x, train_data_y), (dev_data_x, dev_data_y), (test_data_x, test_data_y), vocab_size

class JointLearningDataPreprocess(DataPreprocess):
    """Intent detection & Slot Filling Joint-Learning"""
    def __init__(self):
        super(JointLearningDataPreprocess, self).__init__()

    def load_corpus(self, config):
        train_data_file = config.corpus_info.train_data_file
        print("Loading train data from {} ...".format(train_data_file))
        train_json_data = self.load_json_corpus(train_data_file)
        train_data = [(item["query"], item["intent"], item["tags"]) for item in train_json_data]

        dev_data_file = config.corpus_info.dev_data_file
        print("Loading dev data from {} ...".format(dev_data_file))
        dev_json_data = self.load_json_corpus(dev_data_file)
        dev_data = [(item["query"], item["intent"], item["tags"]) for item in dev_json_data]

        test_data_file = config.corpus_info.test_data_file
        print("Loading test data from {} ...".format(test_data_file))
        test_json_data = self.load_json_corpus(test_data_file)
        test_data = [(item["query"], item["intent"], item["tags"]) for item in test_json_data]

        return train_data, dev_data, test_data

    def __call__(self, config):
        self.make_output_dir(config)
        train_data, dev_data, test_data = self.load_corpus(config)

        vocab, vocab_size = self.load_albert_vocab(config)
        max_seq_len = config.encoder_info.max_seq_len

        train_data_x = encode_query(train_data, vocab, length=max_seq_len)
        dev_data_x = encode_query(dev_data, vocab, length=max_seq_len)
        test_data_x = encode_query(test_data, vocab, length=max_seq_len)

        intent_id_dic, id_intent_dic = self.generateLabelMap(train_data, label_colunm=1)
        intent_id_dic_file = os.path.join(self.base_path, self.intent_id_dic_file)
        id_intent_dic_file = os.path.join(self.base_path, self.id_intent_dic_file)
        
        json.dump(intent_id_dic, open(intent_id_dic_file, "w"), ensure_ascii=False)
        json.dump(id_intent_dic, open(id_intent_dic_file, "w"), ensure_ascii=False)

        train_data_intent = encodeLabel(train_data, intent_id_dic, label_colunm=1)
        dev_data_intent = encodeLabel(dev_data, intent_id_dic, label_colunm=1)
        test_data_intent = encodeLabel(test_data, intent_id_dic, label_colunm=1)

        tag_id_dic, id_tag_dic = self.generateTagsMap(train_data, label_colunm=2)
        tag_id_dic_file = os.path.join(self.base_path, self.tag_id_dic_file)
        id_tag_dic_file = os.path.join(self.base_path, self.id_tag_dic_file)
        
        json.dump(tag_id_dic, open(tag_id_dic_file, "w"), ensure_ascii=False)
        json.dump(id_tag_dic, open(id_tag_dic_file, "w"), ensure_ascii=False)

        train_data_tags = encodeTags(train_data, tag_id_dic, length=max_seq_len, label_colunm=2)
        dev_data_tags = encodeTags(dev_data, tag_id_dic, length=max_seq_len, label_colunm=2)
        test_data_tags = encodeTags(test_data, tag_id_dic, length=max_seq_len, label_colunm=2)

        train = (train_data_x, train_data_intent, train_data_tags)
        dev = (dev_data_x, dev_data_intent, dev_data_tags)
        test = (test_data_x, test_data_intent, test_data_tags)

        return train, dev, test, vocab_size

class TextMatchDataPreprocess(DataPreprocess):
    def __init__(self):
        super(TextMatchDataPreprocess, self).__init__()

    def load_corpus(self, config):
        train_data_file = config.corpus_info.train_data_file
        print("Loading train data from {} ...".format(train_data_file))
        train_json_data = self.load_json_corpus(train_data_file)
        train_data = [(item["query"])for item in train_json_data]

        dev_data_file = config.corpus_info.dev_data_file
        print("Loading dev data from {} ...".format(dev_data_file))
        dev_json_data = self.load_json_corpus(dev_data_file)
        dev_data = [(item["query"]) for item in dev_json_data]

        test_data_file = config.corpus_info.test_data_file
        print("Loading test data from {} ...".format(test_data_file))
        test_json_data = self.load_json_corpus(test_data_file)
        test_data = [(item["query"]) for item in test_json_data]

        return train_data, dev_data, test_data

    def __call__(self, config):
        self.make_output_dir(config)
        train_data, dev_data, test_data = self.load_corpus(config)

        vocab, vocab_size = self.load_albert_vocab(config)
        max_seq_len = config.encoder_info.max_seq_len

        train_data_x = encode_query(train_data, vocab, length=max_seq_len)
        dev_data_x = encode_query(dev_data, vocab, length=max_seq_len)
        test_data_x = encode_query(test_data, vocab, length=max_seq_len)

        # 无监督训练不需要使用标签y
        train_data_y = np.zeros(shape=train_data_x.shape[0])
        dev_data_y = np.zeros(shape=dev_data_x.shape[0])
        test_data_y = np.zeros(shape=test_data_x.shape[0])

        return (train_data_x, train_data_y), (dev_data_x, dev_data_y), (test_data_x, test_data_y), vocab_size

if __name__ == "__main__":
    config = ConfigParser(config_file='../config/app_intent_config')
    #_ = Preprocess()(config)