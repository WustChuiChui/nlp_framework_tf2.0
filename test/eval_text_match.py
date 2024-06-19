import json
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr

import sys
sys.path.append("./")

from config.config_parser import ConfigParser
from trainer.trainer import TextMatchTrainer
# from utils.io_utils import load_corpus
from utils.tf_utils import encode_query_v2

# TEST_FILE_NAME = ...


def load_model(config):
    trainer = TextMatchTrainer(config)
    trainer.load_model()

    return trainer


def load_json_corpus(file_name):
    queries = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            record = json.loads(line.strip())
            queries.append((record["query"]))
    return queries


def load_json_vocab_dic(file_name):
    with open(file_name, "r") as file:
        for line in file.readlines():  # 就一行
            return json.loads(line.strip())


def get_filtered_labels(pred_y, scores, threshold, default):
    pred_labels = []
    for idx in range(len(pred_y)):
        if scores[idx] >= threshold:
            pred_labels.append(pred_y[idx])
        else:
            pred_labels.append(default)
    return pred_labels


config = ConfigParser(config_file='./config/app_text_match_config')

work_result_dir = "{}/{}/{}/{}".format(config.trainer_info.results_dir, config.task_info.name,
                                           config.task_info.task, config.task_info.version)
trainer = load_model(config)
# test_data = load_json_corpus(config.corpus_info.test_data_file)
test_data = [
    "想喝奶茶！",
    "播放周杰伦的《花海》",
    "能否介绍一下西湖的推荐景点？",
    "能否简要列举一下西湖的必看景点？"
]
test_data = (test_data, [])  # 数据与函数对齐
vocab = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "vocab.json"))

test_data_x = encode_query_v2(test_data, vocab, length=config.encoder_info.max_seq_len)
print(test_data_x.shape)
print("Begin to predict...")
result = trainer.model.predict(test_data_x)
print(result.shape[0])
print(result[0])  # 检查是否有0

# corrs = [spearmanr(result[i], result[j]) for i in range(len(result)) for j in range(i+1, len(result))]
for i in range(4):
    for j in range(i+1, 4):
        corr = spearmanr(result[i], result[j])
        print(f"文本“{test_data[0][i]}”和文本“{test_data[0][j]}”的相似度系数为: {corr}")
