from dataset import Dataset

class NeuralNetwork(object):

	def __init__(self, pathDataset):
		self.iD = Dataset(pathProcessed=pathDataset)
		

if __name__ == '__main__':
	pathProcessed = "/home/joan/repository/PURESTYLE/"
	iD = NeuralNetwork(pathProcessed)