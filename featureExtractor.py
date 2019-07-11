from spacyTree import SpacyTree
import spacy
import numpy as np

class FeatureExtractor:

	def __init__(self, text):
		self.raw = text
		nlp = spacy.load('en_core_web_md')
		doc = nlp(text)
		self.trees = []
		for sent in doc.sents:
			iST = SpacyTree(nlp, sent)
			self.trees.append(iST)

		self.getFeatureVectors()


	def getFeatureVectors(self):
		for tree in self.trees:
			vectors = tree.getFeatureVectors()
			for vector in vectors:
				vectorNParent = self.normalizeVector(vector[0])
				vectorNNode = self.normalizeVector(vector[1])
				vectorNChildren = []
				for child in vector[2]:
					vectorNChildren.append(self.normalizeVector(child))

	def normalizeVector(self, vector):
		morphKeys = ["Degree", "Tense", "Person", "VerbForm", "PunctType", "Number", "PronType"]
		if vector:
			### POS[10]  DEP[10]  MORPH[N] HEAD[1] LEVEL[1] OOV(1) ###
			pass




		else:
			return None

class LookUp:

	def __init__(self):
		self.indexDeps = {}
		self.indexPos = {}
		self.indexMorph = {}


if __name__ == '__main__':
	text = "Newman declared abruptly and firmly that he knew nothing about tables and chairs, and that he would accept, in the way of a lodging, with his eyes shut, anything that Tristram should offer him."
	iF = FeatureExtractor(text)
