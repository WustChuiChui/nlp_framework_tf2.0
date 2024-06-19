from config.config_parser import ConfigParser
import json
import tensorflow as tf
import numpy as np
from trainer.trainer import StudentMatchTrainer
from utils.io_utils import load_corpus
from utils.tf_utils import encode_query_v2
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_class_report

T = 0.2
DEFAULT_DOMAIN = "other"
TEST_FILE_NAME = "/Users/wangjia/nlp_task_corpus/data/ar_domain/orgin_ar_test_set.csv"

def load_model(config):
    trainer = StudentMatchTrainer(config)
    trainer.load_model()

    return trainer

def load_json_vocab_dic(file_name):
    with open(file_name, "r") as file:
        for line in file.readlines():
            return json.loads(line.strip())

def get_filtered_labels(pred_y, scores):
    pred_labels = []
    for idx in range(len(pred_y)):
        if scores[idx] >= T:
            pred_labels.append(pred_y[idx])
        else:
            pred_labels.append(DEFAULT_DOMAIN)
    return pred_labels

config = ConfigParser(config_file='./config/ar_domain_student_intent_config')
print(config)

work_result_dir = "{}/{}/{}/{}".format(config.trainer_info.results_dir, config.task_info.name, config.task_info.task, config.task_info.version)
print(work_result_dir)

trainer = load_model(config)
test_data = load_corpus(TEST_FILE_NAME)

label_dic = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "id_intent_json"))
vocab = load_json_vocab_dic(file_name="{}/{}".format(work_result_dir, "vocab.json"))

test_data_x = encode_query_v2(test_data, vocab, length=config.encoder_info.max_seq_len)
logits = trainer.model.predict(test_data_x)

#knowledge_matrix = np.array(trainer.teacher_knowledge.tolist(), dtype=np.float32)
#sim_matrix = tf.reduce_sum(logits[:, tf.newaxis] * knowledge_matrix, axis=-1)
#sim_matrix /= tf.norm(logits[:, tf.newaxis], axis=-1) * tf.norm(knowledge_matrix, axis=-1)

sim_matrix = logits
pred_label = tf.argmax(sim_matrix, axis=1).numpy().tolist()
pred_score = tf.reduce_max(sim_matrix, axis=1).numpy().tolist()
pred_label = [label_dic[str(id)] for id in pred_label]

pred_label = get_filtered_labels(pred_label, pred_score)
print(classification_report(test_data[1], pred_label))

query_set = set()
with open("./results/ar_domain_student_sim_match_test_result.csv", "w") as file:
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