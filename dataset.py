import os
from featureExtractor import FeatureExtractor
import spacy
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Dataset:
	
	def __init__(self, pathRaw=None, pathProcessed=None, maxLen=30):
		self.dataset = []
		self.labelDict = {}
		labelIdx = 0

		if not pathProcessed:
			print("Loading spacy")
			nlp = spacy.load('en_core_web_md')
			print("loaded")
			
			#minWords = 9999
			#maxWords = 0
			#avgWords = 0
			
			listFiles = os.listdir(pathRaw)
			nTexts = len(listFiles)
			fdOutLabels = open("labels.txt","w")
			fdOutCorrespondence = open("labelCorrespondence.txt","w")
			
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
					if len(instanceVectors) == maxLen:
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
			np.save("vectors.npy",self.dataset)

			fdOutCorrespondence.write(str(self.labelDict))
			fdOutCorrespondence.close()
			'''
			print("dataset has",nTexts,"instances")
			print("maxWords",maxWords)
			print("minWords", minWords)
			print("avgWords",avgWords/nTexts)
			'''
		else:
			self.dataset = np.load(pathProcessed+"vectors.npy")
			self.labels = open(pathProcessed+"labels.txt","r").read().split("\n")
			self.labelDict = eval(open(pathProcessed+"labelCorrespondence.txt","r").read())
			print(len(self.dataset[0]))

if __name__ == '__main__':
	path = "/home/joan/Escritorio/Datasets/KaggleCompetition/clean_train/"
	pathProcessed = "/home/joan/repository/PURESTYLE/"
	iD = Dataset(pathProcessed=pathProcessed)