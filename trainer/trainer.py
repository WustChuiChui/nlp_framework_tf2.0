import os, random
from abc import abstractmethod
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from encoder.encoder import *
from utils.io_utils import checkout_dir
from utils.tf_utils import fill_feed_batch_data, fill_feed_batch_data_v2
from common.const import ENCODER_MODULE
from trainer.simcse_loss import *  # 自定义损失函数
import json
import numpy as np

class Trainer:
    def __init__(self, config):
        self.config = config
        self.callback_list = []
        self.work_result_dir = "{}/{}/{}/{}".format(config.trainer_info.results_dir, config.task_info.name, config.task_info.task, config.task_info.version)
        self.saved_model_path = "{}/{}".format(self.work_result_dir, config.trainer_info.saved_model_path)
        self.checkpoint_dir = os.path.dirname("{}/{}".format(self.work_result_dir, self.config.trainer_info.checkpoint_path))

        self.create_model()

    def create_model(self):
        encoder_info = self.config.encoder_info
        self.model = getattr(sys.modules[ENCODER_MODULE], encoder_info.encoder)(encoder_info)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.trainer_info.learning_rate)

        self.model.build_graph(input_shape=(None, encoder_info.max_seq_len))
        self.model.summary()

    def get_callback(self, monitor='val_accuracy'):
        callback_list = []

        callback_list.append(EarlyStopping(monitor=monitor, patience=7, mode='max'))
        checkpoint_path = "{}/{}".format(self.work_result_dir, self.config.trainer_info.checkpoint_path)
        tensorboard_log_dir = "{}/{}".format(self.work_result_dir, "log")

        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor=monitor,
                                          mode='max',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=2)
            callback_list.append(cp_callback)

        checkout_dir(tensorboard_log_dir, do_delete=True)
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        callback_list.append(tensorboard_callback)

        self.callback_list = callback_list

    def save_steps(self):
        self.model.compute_output_shape(input_shape=(None, self.config.encoder_info.max_seq_len))
        tf.saved_model.save(self.model, self.saved_model_path)

    @abstractmethod
    def dev_steps(self, data):
        pass

    @abstractmethod
    def train_steps(self, train_data, val_data):
        pass

    def run(self, train_data, dev_data, test_data):
        if self.config.trainer_info.mode == "train":
            self.train_steps(train_data, dev_data)
            acc, loss = self.dev_steps(test_data)
            print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
            self.save_steps()
        elif self.config.trainer_info.mode == "test":
            acc, loss = self.dev_steps(test_data)
            print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        elif self.config.trainer_info.mode == "save_model":
            self.save_steps()
        else:
            print("config.trainer_info.mode is not provided.")
            exit(0)

class ClassifyTrainer(Trainer):
    def __init__(self, config):
        super(ClassifyTrainer, self).__init__(config)

    def create_model(self):
        print("Build Classify Trainer:")
        super(ClassifyTrainer, self).create_model()

        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        # 若需替换成focal_loss，直接下载对应的库后替换loss即可

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       batch_size=self.config.trainer_info.batch_size,
                       epochs=self.config.trainer_info.epochs,
                       verbose=2,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self):
        print(self.checkpoint_dir)
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)

    def train_steps(self, train_data, val_data):
        self.get_callback()
        self.fit(x_train=train_data[0], y_train=train_data[1], x_val=val_data[0], y_val=val_data[1])
        print('val for training...')
        val_score = self.model.evaluate(val_data[0], val_data[1], batch_size=self.config.trainer_info.batch_size)
        print("val loss:", val_score[0], "val accuracy", val_score[1])

    def dev_steps(self, test_data):
        self.load_model()
        loss, acc = self.model.evaluate(test_data[0], test_data[1], verbose=2)
        result = self.model.predict(test_data[0])
        print(result)
        return acc, loss

class NerTrainerV2(Trainer):
    def __init__(self, config):
        super(NerTrainerV2, self).__init__(config)

    def create_model(self):
        print("Build Ner Trainer:")
        super(NerTrainerV2, self).create_model()

        self.model.compile(optimizer=self.optimizer, loss=self.model.crf.get_loss, metrics=[self.model.crf.get_accuracy])

    def fit(self, x_train, y_train, x_val, y_val):
        print("Train for NerTrainV2....")
        history = self.model.fit(x_train, y_train,
                       batch_size=self.config.trainer_info.batch_size,
                       epochs=self.config.trainer_info.epochs,
                       verbose=2,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)

    def train_steps(self, train_data, val_data):
        self.get_callback(monitor="val_get_accuracy")
        self.fit(x_train=train_data[0], y_train=train_data[1], x_val=val_data[0], y_val=val_data[1])
        print('val for training...')

        pred_y = self.model.predict(val_data[0])
        acc = self.model.crf.get_accuracy(val_data[1], pred_y)
        print("dev accuracy: ", float(acc))
        #val_score = self.model.evaluate(val_data[0], val_data[1], batch_size=self.config.trainer_info.batch_size)
        #print("val loss:", val_score[0], "val accuracy", val_score[1])

    def dev_steps(self, test_data):
        self.load_model()
        pred_y = self.model.predict(test_data[0])
        acc = float(self.model.crf.get_accuracy(test_data[1], pred_y))
        #loss = float(self.model.crf.get_loss(test_data[1], pred_y))

        print("test accuracy: ", acc)

        return acc, 0.1
        #loss, acc = self.model.evaluate(test_data[0], test_data[1], verbose=2)
        #result = self.model.predict(test_data[0])
        #print(result)
        #return acc, loss


class StudentMatchTrainer(Trainer):
    def __init__(self, config):
        super(StudentMatchTrainer, self).__init__(config)

        self.teacher_knowledge = self.load_teacher_knowledge()

    def load_teacher_knowledge(self):
        feature_dic = {}
        with open(self.config.corpus_info.intent_feature_json, "r") as file:
            for line in file.readlines():
                feature_dic = json.loads(line.strip())
        feature_dic = sorted(feature_dic.items(), key=lambda x: x[0])
        teacher_knowledge = [item[1] for item in feature_dic]
        return np.array(teacher_knowledge)

    def create_model(self):
        print("Build Student Match Trainer.")
        super(StudentMatchTrainer, self).create_model()

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, checkpoint_name="model.ckpt", max_to_keep=3)

    def load_model(self):
        self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def tf_cosine_similarity(self, tensor1, tensor2):
        tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1), axis=1))
        tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2), axis=1))

        tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=1)
        cos_sim = tensor1_tensor2 / (tensor1_norm * tensor2_norm)

        return cos_sim

    def sim_hard_sampling(self, logits, candicate_logits, labels, sim_hard_idx=[-4, -3, -2]):
        sim_matrix = tf.reduce_sum(logits[:, tf.newaxis] * candicate_logits, axis=-1)
        norm_candicate_logits = np.array(candicate_logits.tolist(), dtype=np.float32)
        sim_matrix /= tf.norm(logits[:, tf.newaxis], axis=-1) * tf.norm(norm_candicate_logits, axis=-1)

        sim_idx = np.argsort(sim_matrix, axis=-1)
        nag_idx = [sim_idx[i][random.choice(sim_hard_idx)] if sim_idx[i][-1] == labels[i] else sim_idx[i][-1] for i in range(len(sim_idx))]

        nag_logits = np.array([candicate_logits[idx] for idx in nag_idx], dtype=np.float32)

        return nag_logits

    def get_triple_loss(self, logits, labels, lamda=0.8):
        """
        :param logits: [batch_size, num_class]
        :param labels: [batch_size] idx of intent id
        :param teacher_knowledge: [num_class, num_class] learning from Teacher model, semantic features for intent
        :param lamda: threshold for similarity loss
        :return: loss
        """

        # sample pair generated
        pos_logits = np.array([self.teacher_knowledge[idx] for idx in labels], dtype=np.float32)
        #random sampling
        #num_class = len(self.teacher_knowledge)
        #nag_logits = np.array([self.teacher_knowledge[(idx + random.choice(range(1, num_class, 1))) % num_class] for idx in labels], dtype=np.float32)
        #sim_hard sampling
        nag_logits = self.sim_hard_sampling(logits, self.teacher_knowledge, labels)
        # similarity
        pos_cos_sim = self.tf_cosine_similarity(logits, pos_logits)
        nag_cos_sim = self.tf_cosine_similarity(logits, nag_logits)

        # loss
        sim_loss = tf.reduce_mean(tf.math.maximum(0.0, nag_cos_sim + lamda - pos_cos_sim))

        loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
        cross_loss = loss_func(labels, logits)

        loss = sim_loss + cross_loss
        return loss

    def train_one_step(self, text_batch, label_batch):
        with tf.GradientTape() as tape:
            logits = self.model(text_batch, training=True)
            loss = self.get_triple_loss(logits, label_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits

    def get_acc_one_step(self, logits, labels):
        """
        #Similarities
        knowledge_matrix = np.array(self.teacher_knowledge.tolist(), dtype=np.float32)
        sim_matrix = tf.reduce_sum(logits[:, tf.newaxis] * knowledge_matrix, axis=-1)
        sim_matrix /= tf.norm(logits[:, tf.newaxis], axis=-1) * tf.norm(knowledge_matrix, axis=-1)

        pred_y = tf.argmax(sim_matrix, axis=1)
        acc = np.sum(np.equal(pred_y, labels)) / len(labels)
        """
        #cross logits
        acc = np.sum(np.equal(tf.argmax(logits, axis=1), labels)) / len(labels)

        return acc

    def dev_steps(self, dev_data):
        self.load_model()
        logits = self.model(dev_data[0])
        loss = self.get_triple_loss(logits, dev_data[1])
        acc = self.get_acc_one_step(logits, dev_data[1])
        return acc, loss

    def train_steps(self, train_data, dev_data):
        best_acc = 0
        step = 0
        for epoch in range(self.config.trainer_info.epochs):
            for text_batch, labels_batch in fill_feed_batch_data(train_data, self.config.trainer_info.batch_size):
                step = step + 1
                loss, logits = self.train_one_step(text_batch, labels_batch)
                if step % 1000 == 0:
                    accuracy = self.get_acc_one_step(logits, labels_batch)
                    print('epoch %d, training step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            acc, loss = self.dev_steps(dev_data)
            print('epoch %d, dev loss %.4f , dev accuracy %.4f' % (epoch, loss, acc))
            if acc > best_acc:
                best_acc = acc
                self.ckpt_manager.save()

class NerTrainer(Trainer):
    def __init__(self, config):
        super(NerTrainer, self).__init__(config)

    def create_model(self):
        print("Build Ner Trainer:")
        super(NerTrainer, self).create_model()

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, checkpoint_name='model.ckpt', max_to_keep=3)

    def load_model(self):
        self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    @tf.function
    def train_one_step(self, text_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = self.model(text_batch, labels_batch, training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits, text_lens

    def get_acc_one_step(self, logits, text_lens, labels_batch):
        paths = []
        accuracy = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], self.model.transition_params)
            paths.append(viterbi_path)
            correct_prediction = tf.equal(
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'), dtype=tf.int32),
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'), dtype=tf.int32)
            )
            accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = accuracy / len(paths)
        return accuracy

    def dev_steps(self, dev_data):
        self.load_model()
        logits, text_lens, log_likelihood = self.model(dev_data[0], dev_data[1])
        loss = - tf.reduce_mean(log_likelihood)
        acc = self.get_acc_one_step(logits, text_lens, dev_data[1])
        return acc, loss

    def train_steps(self, train_data, dev_data):
        best_acc = 0
        step = 0
        for epoch in range(self.config.trainer_info.epochs):
            for text_batch, labels_batch in fill_feed_batch_data(train_data, self.config.trainer_info.batch_size):
                step = step + 1
                loss, logits, text_lens = self.train_one_step(text_batch, labels_batch)
                if step % 1000 == 0:
                    accuracy = self.get_acc_one_step(logits, text_lens, labels_batch)
                    print('epoch %d, training step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            acc, loss = self.dev_steps(dev_data)
            print('epoch %d, dev loss %.4f , dev accuracy %.4f' % (epoch, loss, acc))
            if acc > best_acc:
                best_acc = acc
                self.ckpt_manager.save()

    def predict(self, test_data):
        """上线时用"""
        self.load_model()
        logits, text_lens = self.model(test_data[0])
        paths = []
        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], self.model.transition_params)
            paths.append(viterbi_path)
        print(test_data[0][0], paths[0])

class JointLearningTrainer(Trainer):
    def __init__(self, config):
        super(JointLearningTrainer, self).__init__(config)

    def create_model(self):
        print("Build JointLearning Trainer")
        super(JointLearningTrainer, self).create_model()

        self.intent_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, checkpoint_name='model.ckpt', max_to_keep=3)

    def load_model(self):
        self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def train_one_step(self, batch_data):
        with tf.GradientTape() as tape:
            classify_logits, ner_logits, text_lens, log_likelihood = self.model(batch_data[0], tags=batch_data[2], training=True)
            ner_loss = - tf.reduce_mean(log_likelihood)
            classify_loss = self.intent_loss(batch_data[1], classify_logits)
            loss = classify_loss + ner_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, classify_logits, ner_logits, text_lens

    def get_ner_acc_one_step(self, logits, text_lens, labels_batch):
        paths = []
        accuracy = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], self.model.transition_params)
            paths.append(viterbi_path)
            correct_prediction = tf.equal(
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'), dtype=tf.int32),
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'), dtype=tf.int32)
            )
            accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = accuracy / len(paths)
        return accuracy

    def train_steps(self, train_data, val_data):
        cla_best_acc, ner_best_acc, step = 0, 0, 0
        for epoch in range(self.config.trainer_info.epochs):
            for batch_data in fill_feed_batch_data_v2(train_data, self.config.trainer_info.batch_size):
                step = step + 1
                loss, classify_logits, ner_logits, text_lens = self.train_one_step(batch_data)
                if step % 200 == 0:
                    cla_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(classify_logits, axis=1), batch_data[1]), tf.float32))
                    ner_acc = self.get_ner_acc_one_step(ner_logits, text_lens, batch_data[2])
                    print('epoch %d, training step %d, loss %.4f , classify acc: %.4f , ner acc: %.4f' % (epoch, step, loss, cla_acc, ner_acc))
            cla_acc, ner_acc, loss = self.dev_steps(val_data)
            print('epoch %d, dev loss %.4f , dev classify acc: %.4f , dev ner acc: %.4f' % (epoch, loss, cla_acc, ner_acc))
            if cla_acc > cla_best_acc and ner_acc > ner_best_acc: #2个任务同时提升
                print("model performance improved, save checkpoint.")
                cla_best_acc = cla_acc
                ner_best_acc = ner_acc
                self.ckpt_manager.save()

    def dev_steps(self, data):
        self.load_model()
        classify_logits, ner_logits, text_lens, log_likelihood = self.model(data[0], tags=data[2])
        ner_loss = - tf.reduce_mean(log_likelihood)
        ner_acc = self.get_ner_acc_one_step(ner_logits, text_lens, data[2])

        cla_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(classify_logits, axis=1), data[1]), tf.float32))
        classify_loss = self.intent_loss(data[1], classify_logits)

        loss = ner_loss + classify_loss

        return cla_acc, ner_acc, loss

    def run(self, train_data, dev_data, test_data):
        if self.config.trainer_info.mode == "train":
            self.train_steps(train_data, dev_data)
            cla_acc, ner_acc, loss = self.dev_steps(test_data)
            print("Restored model, classify accuracy: {:5.2f}, ner accuracy: {:5.2f}%".format(100 * cla_acc, 100 * ner_acc))
            self.save_steps()
        elif self.config.trainer_info.mode == "test":
            cla_acc, ner_acc, loss = self.dev_steps(test_data)
            print("Restored model, classify accuracy: {:5.2f}, ner accuracy: {:5.2f}%".format(100 * cla_acc, 100 * ner_acc))
        elif self.config.trainer_info.mode == "save_model":
            self.save_steps()
        else:
            print("config.trainer_info.mode is not provided.")
            exit(0)

class TextMatchTrainer(Trainer):
    def __init__(self, config):
        super(TextMatchTrainer, self).__init__(config)

    def create_model(self):
        print("Build Text Match Trainer:")
        super(TextMatchTrainer, self).create_model()

        # self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.CosineSimilarity(), metrics=['loss'])
        # self.model.compile(optimizer=self.optimizer, loss=SimLoss, metrics=['loss'])
        self.model.compile(optimizer=self.optimizer, loss=SimHardLoss(), metrics=['mse'])

    def get_callback(self, monitor='val_loss'):
        callback_list = []

        callback_list.append(EarlyStopping(monitor=monitor, patience=7, mode='max'))
        checkpoint_path = "{}/{}".format(self.work_result_dir, self.config.trainer_info.checkpoint_path)
        tensorboard_log_dir = "{}/{}".format(self.work_result_dir, "log")

        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor=monitor,
                                          mode='min',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=1)
            callback_list.append(cp_callback)

        checkout_dir(tensorboard_log_dir, do_delete=True)
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        callback_list.append(tensorboard_callback)

        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       batch_size=self.config.trainer_info.batch_size,
                       epochs=self.config.trainer_info.epochs,
                       verbose=2,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self):
        print(self.checkpoint_dir)
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)

    def train_steps(self, train_data, val_data):
        self.get_callback()
        self.fit(x_train=train_data[0], y_train=train_data[1], x_val=val_data[0], y_val=val_data[1])
        print('val for training...')
        val_score = self.model.evaluate(val_data[0], val_data[1], batch_size=self.config.trainer_info.batch_size)
        print("val loss:", val_score[0], "val accuracy", val_score[1])

    def dev_steps(self, test_data):
        self.load_model()
        loss, acc = self.model.evaluate(test_data[0], test_data[1], verbose=2)
        result = self.model.predict(test_data[0])
        print(result)
        return acc, loss

    def run(self, train_data, dev_data, test_data):
        if self.config.trainer_info.mode == "train":
            self.train_steps(train_data, dev_data)
            acc, loss = self.dev_steps(test_data)
            print("Restored model, loss: {:5.2f}".format(loss))
            self.save_steps()
        elif self.config.trainer_info.mode == "test":
            acc, loss = self.dev_steps(test_data)
            print("Restored model, loss: {:5.2f}".format(loss))
        elif self.config.trainer_info.mode == "save_model":
            self.save_steps()
        else:
            print("config.trainer_info.mode is not provided.")
            exit(0)
