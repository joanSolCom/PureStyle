from dataset import Dataset
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Bidirectional, Input, Conv1D, Flatten
from keras.models import Sequential, Model
from keras import optimizers, regularizers, constraints
from keras.utils import to_categorical
import numpy as np

class NeuralNetwork(object):

	def __init__(self, pathV, pathL, pathC):
		self.iD = Dataset()
		self.iD.loadDataset(pathV, pathL, pathC)

		dataset_lstm = []

		
		inputs = Input(shape=(30,268))
		lstm = LSTM(268)(inputs)
		model = Model(inputs=inputs, outputs=lstm)


		for instance in self.iD.dataset:
			after_lstm = model.predict(instance.reshape(1,30,268))
			dataset_lstm.append(after_lstm)


		dataset_lstm = np.array(dataset_lstm)
		
		'''
		for instance in self.iD.dataset:
			meanVec = np.average(instance, axis=0)
			dataset_lstm.append(meanVec.reshape(1,268))

		dataset_lstm = np.array(dataset_lstm)
		'''

		model = Sequential()
		model.add(LSTM(64))
		model.add(Dense(3,activation='softmax'))

		x_train = dataset_lstm[:16000]
		y_train = to_categorical(self.iD.labels[:16000])

		x_valid = dataset_lstm[16000:17500]
		y_valid = to_categorical(self.iD.labels[16000:17500])

		x_test = dataset_lstm[17500:]
		y_test = to_categorical(self.iD.labels[17500:])

		#sgd = optimizers.SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
		adam = optimizers.Adam()
		history = model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
		nb_epoch = 50
		model.fit(x_train, y_train, epochs=nb_epoch, validation_data=(x_valid, y_valid), batch_size=256)

		scores = model.evaluate(x_test, y_test, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == '__main__':
	pathV = "/home/joan/repository/PURESTYLE/vectors.npy"
	pathL = "/home/joan/repository/PURESTYLE/labels.txt"
	pathC = "/home/joan/repository/PURESTYLE/labelCorrespondence.txt"
	iD = NeuralNetwork(pathV, pathL, pathC)