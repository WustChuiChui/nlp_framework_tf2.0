from config.config_parser import ConfigParser
import json
import tensorflow as tf
from trainer.trainer import ClassifyTrainer
from utils.io_utils import load_corpus
from utils.tf_utils import encode_query_v2
import numpy as np
from sklearn.metrics import classification_report

TEST_FILE_NAME = "/Users/wangjia/nlp_task_corpus/data/ar_domain/orgin_ar_test_set.csv"

def load_model(config):
    trainer = ClassifyTrainer(config)
    trainer.load_model()
    return trainer

def load_json_vocab_dic(file_name):
    with open(file_name, "r") as file:
        for line in file.readlines(): #就一行
            return json.loads(line.strip())

config = ConfigParser(config_file='./config/ar_domain_intent_config')
print(config)

work_result_dir = "{}/{}/{}/{}".format(config.trainer_info.results_dir, config.task_info.name, config.task_info.task, config.task_info.version)

trainer = load_model(config)
test_data = load_corpus(TEST_FILE_NAME)

label_dic = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "id_intent_json"))
label_id_dic = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "intent_id_json"))
vocab = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "vocab.json"))

#test_data = [test_data[0][:200], test_data[1][:200]]
test_data_x = encode_query_v2(test_data, vocab, length=config.encoder_info.max_seq_len)
result = trainer.model.predict(test_data_x)

pred_label = tf.argmax(result, axis=1).numpy().tolist()
pred_score = tf.reduce_max(result, axis=1).numpy().tolist()
pred_label = [label_dic[str(id)] for id in pred_label]

label_feature_dic = {}
for idx in range(len(pred_label)):
    query = test_data[0][idx]
    label = test_data[1][idx]
    pred_y = pred_label[idx]
    if label != pred_y: continue
    #print(query + "\t" + label + "\t" + pred_y)
    if pred_y not in label_feature_dic:
        label_feature_dic[pred_y] = []
    label_feature_dic[pred_y].append(result[idx])

res_dic = {}
for key, logits_list in label_feature_dic.items():
    #print(key, label_id_dic[key], np.argmax(np.array(logits_list).mean(axis=0)))
    res_dic[label_id_dic[key]] = np.array(logits_list).mean(axis=0).tolist()
    #print(np.array(logits_list).mean(axis=0))

print(res_dic)
json.dump(res_dic, open("intent_feature_json", "w"), ensure_ascii=False)
intent_feature_dic = load_json_vocab_dic("intent_feature_json")
intent_feature_dic = sorted(intent_feature_dic.items(), key=lambda x:x[0])
intent_feature_dic = [item[1] for item in intent_feature_dic]
print(intent_feature_dic)
