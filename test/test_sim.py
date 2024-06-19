import tensorflow as tf
import numpy as np

def tf_cosine_similarity(tensor1, tensor2):
     tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1), axis=1))
     tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2), axis=1))

     tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=1)
     cos_sim = tensor1_tensor2 / (tensor1_norm * tensor2_norm)

     return cos_sim


labels = [0, 1, 2, 1, 2]

logits = np.array([[1.0, 0, 0],
     [0.8, 0.1, 0.1],
     [0.1, 0.8, 0.1],
     [0.05, 0.9, 0.05],
     [0.1, 0.1, 0.8]])

t_logits = np.array([[0.9, 0.05, 0.05],
     [0.02, 0.9, 0.08],
     [0.1, 0.05, 0.85]])
#predict
sim = tf.reduce_sum(logits[:, tf.newaxis] * t_logits, axis=-1)
sim /= tf.norm(logits[:, tf.newaxis], axis=-1) * tf.norm(t_logits, axis=-1)

print("sim-mratrics: ")
print(sim)
"""
pred_y = tf.argmax(sim, axis=1)
print(pred_y)
acc = np.sum(np.equal(pred_y, labels)) /len(labels)
print(acc)
exit(0)
"""
print("sim vector instance: ")


num_class = 3
import random

def get_triple_loss(logits, labels, t_logits):
     #sample pair generated
     pos_logits = np.array([t_logits[idx] for idx in labels])
     #nag_logits = np.array([t_logits[(idx + random.choice(range(1, num_class, 1))) % num_class] for idx in labels])
     #print("pos_logits: ", pos_logits)
     nag_logits = sim_hard_sampling(logits=logits, candicate_logits=t_logits, labels=labels)
     #print("nag_logits: ", nag_logits)
     #similarity
     pos_cos_sim = tf_cosine_similarity(logits, pos_logits)
     nag_cos_sim = tf_cosine_similarity(logits, nag_logits)

     #loss
     loss = tf.math.maximum(0.0, nag_cos_sim + 0.5 - pos_cos_sim)
     loss = tf.reduce_mean(loss)

     return loss

def sim_hard_sampling(logits, candicate_logits, labels, nag_idx=-3):
     sim_matrix = tf.reduce_sum(logits[:, tf.newaxis] * candicate_logits, axis=-1)
     sim_matrix /= tf.norm(logits[:, tf.newaxis], axis=-1) * tf.norm(candicate_logits, axis=-1)

     sim_idx = np.argsort(sim_matrix, axis=-1)
     nag_idx = [sim_idx[i][nag_idx] if sim_idx[i][-1] == labels[i] else sim_idx[i][-1] for i in range(len(sim_idx))]

     nag_logits = np.array([candicate_logits[idx] for idx in nag_idx])
     #print("nag_logits: ", nag_logits)
     return nag_logits

loss = get_triple_loss(logits, labels, t_logits)
