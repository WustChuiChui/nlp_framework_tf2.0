import json
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("./")

from config.config_parser import ConfigParser
from trainer.trainer import ClassifyTrainer
# from utils.io_utils import load_corpus
from utils.tf_utils import encode_query_v2
from sklearn.metrics import classification_report


T = 0.7
DEFAULT_DOMAIN = "default_llm"
TEST_FILE_NAME = "data/intent/bad_case.csv"


def load_model(config):
    trainer = ClassifyTrainer(config)
    trainer.load_model()

    return trainer


def load_json_corpus(file_name):
    queries, intents = [], []
    with open(file_name, "r") as file:
        for line in file.readlines():
            record = json.loads(line.strip())
            queries.append(record["query"])
            intents.append(record["intent"])
    return queries, intents


def load_json_vocab_dic(file_name):
    with open(file_name, "r") as file:
        for line in file.readlines(): #就一行
            return json.loads(line.strip())


def get_filtered_labels(pred_y, scores):
    pred_labels = []
    for idx in range(len(pred_y)):
        if scores[idx] >= T:
            pred_labels.append(pred_y[idx])
        else:
            pred_labels.append(DEFAULT_DOMAIN)
    return pred_labels


config = ConfigParser(config_file='./config/app_intent_config')
# config.trainer_info.results_dir = "../results"  # 与相对路径对齐

work_result_dir = "{}/{}/{}/{}".format(config.trainer_info.results_dir, config.task_info.name, config.task_info.task, config.task_info.version)

trainer = load_model(config)
test_data = load_json_corpus(TEST_FILE_NAME)

label_dic = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "id_intent_json"))
vocab = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "vocab.json"))

test_data_x = encode_query_v2(test_data, vocab, length=config.encoder_info.max_seq_len)

print("begin to predict")
result = trainer.model.predict(test_data_x)
pred_label = tf.argmax(result, axis=1).numpy().tolist()
pred_score = tf.reduce_max(result, axis=1).numpy().tolist()
pred_label = [label_dic[str(id)] for id in pred_label]

pred_label = get_filtered_labels(pred_label, pred_score)
print(classification_report(test_data[1], pred_label))

query_set = set()
with open("./results/intent_test_result.csv", "w") as file:
    file.writelines("query\tlabel\tpred\tscore\n")
    for i in range(len(pred_label)):
        query = test_data[0][i]
        label = test_data[1][i]
        y_pred = pred_label[i]
        score = pred_score[i]
        if query in query_set: continue
        query_set.add(query)
        if label != y_pred:
            file.writelines(query + "\t" + label + "\t" + y_pred + "\t" + str(round(score, 3)) + "\n")