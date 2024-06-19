import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from common.crf import CRF

class MyBiLSTMCRF(Model):
    def __init__(self, vocabSize, maxLen, tagIndexDict, tagSum, sequenceLengths=None, vecSize=100):
        super(MyBiLSTMCRF, self).__init__()

        self.vocabSize = vocabSize
        self.vecSize = vecSize
        self.maxLen = maxLen
        self.tagSum = tagSum
        self.sequenceLengths = sequenceLengths
        self.tagIndexDict = tagIndexDict

        self.buildBiLSTMCRF()

    def getTransParam(self, y):
        self.trainY = y
        yList = self.trainY.tolist()
        transParam = np.zeros([self.tagSum, self.tagSum])
        for rowI in range(len(yList)):

            for colI in range(len(yList[rowI]) - 1):
                transParam[yList[rowI][colI]][yList[rowI][colI + 1]] += 1
        for rowI in range(transParam.shape[0]):
            transParam[rowI] = transParam[rowI] / np.sum(transParam[rowI])
        return transParam

    def buildBiLSTMCRF(self):

        myModel = Sequential()
        myModel.add(tf.keras.layers.Input(shape=(self.maxLen,)))
        myModel.add(tf.keras.layers.Embedding(self.vocabSize, self.vecSize))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tagSum, return_sequences=True, activation="tanh"), merge_mode='sum'))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tagSum, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf = CRF(self.tagSum, name='crf_layer')
        myModel.add(crf)
        myModel.compile('adam', loss={'crf_layer': crf.get_loss})
        myModel.summary()
        self.myBiLSTMCRF = myModel

    def fit(self, X, y, epochs=100, transParam=None):
        if len(y.shape) == 3:
            y = np.argmax(y, axis=-1)
        if self.sequenceLengths is None:
            self.sequenceLengths = [row.shape[0] for row in y]

        #self.transParam = self.getTransParam(y, self.tagIndexDict)

        history = self.myBiLSTMCRF.fit(X, y, epochs=epochs, callbacks=[TensorBoard(log_dir="./logs", histogram_freq=1)])

        return history

    def predict(self, X):
        preYArr = self.myBiLSTMCRF.predict(X)
        return preYArr


if __name__ == "__main__":
    tagIndexDict = {"O": 0, "B_APP": 1, "I_APP": 2, "E_APP": 3}
    myModel = MyBiLSTMCRF(vocabSize=20000, maxLen=10, tagIndexDict=tagIndexDict, tagSum=4, sequenceLengths=None)
    X = np.array([[ 1,  2,  3,  4,  5,  6,  0,  0,  0,  0],
                 [ 1,  2,  7,  8,  9,  4, 10,  6,  0,  0],
                 [ 1,  1,  2,  8,  9,  4, 11,  6,  0,  0],
                 [ 2,  4,  1,  2,  6,  0,  0,  0,  0,  0],
                 [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0],
                 [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0]])
    Y = np.array([[ 1,  2,  3,  0,  0,  0,  0,  0,  0,  0],
                 [ 1,  2,  2,  3,  0,  0, 0,  0,  0,  0],
                 [ 1,  2,  2,  3,  0,  0, 0,  0,  0,  0],
                 [ 0,  0,  1,  2,  2,  2,  3,  0,  0,  0],
                 [ 0,  0,  0,  1,  2,  3,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  1,  2,  2,  2,  3,  0]])
    history = myModel.fit(X, Y, epochs=5)
    myModel.compute_output_shape(input_shape=(None, 10))
    tf.saved_model.save(myModel, "./test_crf")
    pred_y = myModel.predict([[3,4,5, 6,7,8,9,0, 2, 4]])
    print(pred_y)