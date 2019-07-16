import os
from featureExtractor import FeatureExtractor
import spacy
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Dataset:
	
	def __init__(self, maxLen=30):
		self.dataset = []
		self.labelDict = {}
		self.maxLen = maxLen

	def createDataset(self, pathRaw, pathLabels, pathVectors, pathCorrespondence):
		
		print("Loading spacy")
		nlp = spacy.load('en_core_web_md')
		print("loaded")
		labelIdx = 0

		#minWords = 9999
		#maxWords = 0
		#avgWords = 0
		
		listFiles = os.listdir(pathRaw)
		nTexts = len(listFiles)
		fdOutLabels = open(pathLabels,"w")
		fdOutCorrespondence = open(pathCorrespondence,"w")
		
		for i, fname in enumerate(listFiles):
			label = fname.split("_")[1]
			if label not in self.labelDict:
				self.labelDict[label] = labelIdx
				labelIdx+=1

			numeric_label = self.labelDict[label]
			fdOutLabels.write(str(numeric_label)+"\n")

			fd = open(pathRaw+fname,"r")
			raw = fd.read()
			iF = FeatureExtractor(raw, nlp)
			#FIRST SENTENCE ONLY NOW
			instanceVectors = []
			for wordDict in iF.features[0]:
				instanceVectors.append(wordDict["vector"])
				#only include maxWords vectors
				if len(instanceVectors) == self.maxLen:
					break

			'''
			nWords = len(instanceVectors)
			if nWords > maxWords:
				maxWords = nWords
			if nWords < minWords:
				minWords = nWords

			avgWords += nWords
			
			'''
			
			self.dataset.append(instanceVectors)
			fd.close()
			print(i, "of", nTexts)

		padded_vectors = pad_sequences(self.dataset)

		fdOutLabels.close()
		self.dataset = np.array(padded_vectors)
		#save feature vectors per text
		np.save(pathVectors,self.dataset)

		fdOutCorrespondence.write(str(self.labelDict))
		fdOutCorrespondence.close()
		'''
		print("dataset has",nTexts,"instances")
		print("maxWords",maxWords)
		print("minWords", minWords)
		print("avgWords",avgWords/nTexts)
		'''
		

	def loadDataset(self, pathVectors, pathLabels, pathCorrespondence):
		self.dataset = np.load(pathVectors)
		self.labels = open(pathLabels,"r").read().strip().split("\n")
		for i, label in enumerate(self.labels):
			self.labels[i] = int(label)

		self.labelDict = eval(open(pathCorrespondence,"r").read())
		print("HOLA")

if __name__ == '__main__':
	path = "/home/joan/Escritorio/Datasets/KaggleCompetition/clean_train/"
	pathV = "/home/joan/repository/PURESTYLE/vectors.npy"
	pathL = "/home/joan/repository/PURESTYLE/labels.txt"
	pathC = "/home/joan/repository/PURESTYLE/labelCorrespondence.txt"

	iD = Dataset()
	#iD.createDataset(path, pathL, pathV, pathC)
	iD.loadDataset(pathV, pathL, pathC)