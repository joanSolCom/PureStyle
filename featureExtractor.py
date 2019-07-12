from spacyTree import SpacyTree
import spacy
import numpy as np

class FeatureExtractor:

	def __init__(self, text, nlp):
		self.raw = text
		doc = nlp(text)
		self.doc = doc

		self.trees = []
		for sent in doc.sents:
			iST = SpacyTree(nlp, sent)
			self.trees.append(iST)

		self.getFeatureVectors()
		self.getSentenceFeatures()

	def getFeatureVectors(self):
		self.lookup = LookUp()
		textVector = []

		for tree in self.trees:
			vectors = tree.getFeatureVectors()
			sentenceVector = []
			for vector in vectors:
				wordDictVector = {}
				aggregatedVector = []
				
				vectorNParent = self.normalizeVector(vector[0])
				aggregatedVector.extend(vectorNParent)
				
				vectorNNode = self.normalizeVector(vector[1])
				aggregatedVector.extend(vectorNNode)
				wordDictVector["vector"] = aggregatedVector

				vectorNChildren = []
				for child in vector[2]:
					vectorNChildren.append(self.normalizeVector(child))

				wordDictVector["childrenVectors"] = vectorNChildren

				sentenceVector.append(wordDictVector)
			textVector.append(sentenceVector)

		'''
			[0]sent1:
				[0][0]w1
					vector -> PARENTVECTOR NODEVECTOR
					childrenvectors:[]
				[0][1]w2
					"vector" -> PARENTVECTOR NODEVECTOR
					"childrenVectors":[]
				...

		'''

		self.features = textVector

	def getSentenceFeatures(self):
		self.sentenceFeatures = []
		for tree in self.trees:
			self.sentenceFeatures.append(tree.getFeatures())

	def normalizeVector(self, vector):
		morphKeys = ["Poss", "ConjType", "Degree", "Tense", "Person", "Aspect","VerbType" ,"VerbForm", "PunctType", "Number", "PronType"]
		normalizedVector = []

		if vector:
			### POS[10]  DEP[10]  MORPH[11 * 10] ID[1] PARENT[1] DEPTH[1] OOV[1] ###
			posNorm = self.lookup.getPos(vector["pos"])
			depNorm = self.lookup.getDep(vector["dependency"])

			morphNorm = []
			for key in morphKeys:
				if key in vector["morph"]:
					morphNorm.extend(self.lookup.getMorph(vector["morph"][key]))
				else:
					morphNorm.extend(np.zeros(10))

			normalizedVector.extend(posNorm)
			normalizedVector.extend(depNorm)
			normalizedVector.extend(morphNorm)
			normalizedVector.append(vector["id"])
			normalizedVector.append(vector["parent"])
			normalizedVector.append(vector["depth"])
			if vector["oov"]:
				normalizedVector.append(1)
			else:
				normalizedVector.append(0)
		else:
			normalizedVector = np.zeros(134)

		return normalizedVector


class LookUp:

	POS_DIMENSIONALITY = 10
	MORPH_DIMENSIONALITY = 10
	DEP_DIMENSIONALITY = 10

	def __init__(self):
		self.indexDep = {}
		self.indexPos = {}
		self.indexMorph = {}

	def getPos(self, key):
		if key not in self.indexPos:
			randArray = np.random.uniform(-1,1,size=self.POS_DIMENSIONALITY)
			self.indexPos[key] = randArray
		
		return self.indexPos[key]

	def getDep(self, key):
		if key not in self.indexDep:
			randArray = np.random.uniform(-1,1,size=self.DEP_DIMENSIONALITY)
			self.indexDep[key] = randArray
		
		return self.indexDep[key]

	def getMorph(self, key):
		if key not in self.indexMorph:
			randArray = np.random.uniform(-1,1,size=self.MORPH_DIMENSIONALITY)
			self.indexMorph[key] = randArray
		
		return self.indexMorph[key]



if __name__ == '__main__':
	nlp = spacy.load('en_core_web_md')
	text = "Newman declared abruptly and firmly that he knew nothing about tables and chairs, and that he would accept, in the way of a lodging, with his eyes shut, anything that Tristram should offer him."
	iF = FeatureExtractor(text, nlp)
