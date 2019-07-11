import spacy

class SpacyTree():

	def __init__(self, nlp, sent):
		self.sent = sent
		self.nlp = nlp
		self.node_dict = {}
		self.root = None
		self.buildTree()
		self.extractFeatures()

	def buildTree(self):
		for token in self.sent:
			n = Node(token, self.nlp)
			self.node_dict[n.id] = n
			if token.dep_ == "ROOT":
				self.root = n

		for idNode, iNode in self.node_dict.items():
			if iNode.meta.head.i != iNode.id:
				self.node_dict[iNode.meta.head.i].children.append(iNode)
				self.node_dict[iNode.id].parent = self.node_dict[iNode.meta.head.i]

		for idNode, iNode in self.node_dict.items():
			depth = self.getNodeDepth(iNode)
			iNode.depth = depth
			#print(iNode)

		#print(self)

	def getWidthIterator(self, initNode = None):
		
		if not initNode:
			initNode = self.root

		queue = []
		queue.append(initNode)

		while queue:
			current = queue.pop()
			if current:
				yield current
				for child in current.children:
					queue.insert(0,child)

	def getDepthIterator(self, initNode = None):
		
		if not initNode:
			initNode = self.root
		
		stack = []
		stack.append(initNode)

		while stack:
			current = stack.pop(0)
			if current:
				yield current
				for child in current.children:
					stack.insert(0,child)

	def extractFeatures(self):
		self.ramificationFactor = self.getRamificationFactor()
		self.maxWidth = self.getMaxWidth()
		self.maxDepth = self.getMaxDepth()

	def getRamificationFactor(self, initNode = None):
		if initNode:
			it = self.getWidthIterator(initNode)
		else:
			it = self.getWidthIterator()

		acumChilds = 0
		levels = 1
		for current in it:
			nchilds = len(current.children)
			if nchilds > 0:
				acumChilds+=nchilds
				levels+=1

		return acumChilds / levels

	def getMaxWidth(self, initNode = None):
		it = self.getWidthIterator(initNode)
		maxWidth = 0

		for current in it:
			nchilds = len(current.children)
			if nchilds > maxWidth:
				maxWidth = nchilds

		return maxWidth

	def getMaxDepth(self, initNode = None):
		if not initNode:
			initNode = self.root

		return self.getMaxDepthRecursive(initNode)


	def getMaxDepthRecursive(self, node):
		depth = []

		if node:
			if not node.children:
				return 0
		if not node:
			return 0
		
		for child in node.children:
			depth.append(self.getMaxDepthRecursive(child))

		return 1 + max(depth)

	def getNodeDepth(self, node):
		current = node
		depth = 0
		while current.parent:
			depth+=1
			current = current.parent
		return depth

	def getFeatureVectors(self):
		it = self.getDepthIterator()
		vectors = []
		for node in it:
			vecNode = node.getFeatureVector()
			vecParent = None
			vecChildren = []

			if node.parent:
				vecParent = node.parent.getFeatureVector()

			if node.children:
				vecChildren = []
				for child in node.children:
					vecChildren.append(child.getFeatureVector())
		
			vectors.append([vecParent, vecNode, vecChildren])

		return vectors


	def __str__(self):
		strRepr = ""
		
		queue = []
		queue.append(self.root)
		strRepr+= "Sentence: "+ str(self.sent) + "\n"
		strRepr+="Lemmas:\n"
		for token in self.sent:
			strRepr+=str(token.lemma_)+" "
		strRepr += "\nROOT-> "+ self.root.lemma+"("+str(self.root.id) + ")\n"
		i = 1
		while queue:
			current = queue.pop()
			if current.children:
				strRepr += "CHILDREN OF "+current.lemma+"("+str(current.id)+") -> "
				for child in current.children:
					strRepr += child.lemma+"("+str(child.id) + ")\t"
					queue.insert(0,child)
	
				strRepr +="\n"
			i+=1

		return strRepr

class Node:
	def __init__(self, token, nlp):
		self.meta = token
		self.children = []
		self.id = token.i
		self.parent = None
		self.dependency = token.dep_
		self.pos = token.tag_
		self.lemma = token.lemma_
		self.oov = token.is_oov
		self.depth = None
		self.morph = nlp.vocab.morphology.tag_map[token.tag_]

	def getFeatureVector(self):
		featureVector = {}
		featureVector["id"] = self.id
		featureVector["pos"] = self.pos
		featureVector["depth"] = self.depth
		if self.parent:
			featureVector["parent"] = self.parent.id
		else:
			featureVector["parent"] = -1
		featureVector["morph"] = self.morph
		featureVector["dependency"] = self.dependency
		featureVector["oov"] = self.oov
		return featureVector

	def __str__(self):
		strRepr = ""
		strRepr += str(self.id) + " " + self.lemma + " " + self.dependency + " Depth:" + str(self.depth) +" Morph: "+str(self.morph) + " OOV: "+str(self.oov)
		return strRepr

if __name__ == '__main__':
	en_nlp = spacy.load('en_core_web_md')
	doc = en_nlp("This is an example sentence, a super duper dfkj cool incredible sentence; however, he knows this is not true really.")
	iST = SpacyTree(en_nlp, list(doc.sents)[0])
	it = iST.getDepthIterator()
	print("Depth")
	for node in it:
		print(node)

	print("Width")
	it = iST.getWidthIterator()
	for node in it:
		print(node)