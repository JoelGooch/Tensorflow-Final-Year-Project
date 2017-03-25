
class Layer:
	def __init__(self, layerName):
		self.layerName = layerName

class ConvLayer(Layer):
	def __init__(self, layerName, kernelSize, stride, actFunction, numOutputFilters, weightInit, weightVal, biasInit, biasVal, padding, normalize, dropout , keepRate):
		Layer.__init__(self, layerName)
		self.layerType = 'Convolution'
		self.kernelSize = kernelSize
		self.stride = int(stride)
		self.actFunction = actFunction
		self.numOutputFilters = int(numOutputFilters)
		self.weightInit = weightInit
		self.weightVal = float(weightVal)
		self.biasInit = biasInit
		self.biasVal = float(biasVal)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class MaxPoolingLayer(Layer):
	def __init__(self, layerName, kernelSize, stride, padding, normalize, dropout, keepRate):
		Layer.__init__(self, layerName)
		self.layerType = 'Max Pool'
		self.kernelSize = int(kernelSize)
		self.stride = int(stride)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class DenseLayer(Layer):
	def __init__(self, layerName, actFunction, numOutputNodes, weightInit, weightVal, biasInit, biasVal, normalize, dropout, keepRate):
		Layer.__init__(self, layerName)
		self.layerType = 'Dense'
		self.actFunction = actFunction
		self.numOutputNodes = int(numOutputNodes)
		self.weightInit = weightInit
		self.weightVal = float(weightVal)
		self.biasInit = biasInit
		self.biasVal = float(biasVal)
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class OutputLayer(Layer):
	def __init__(self, layerName, actFunction, weightInit, weightVal, biasInit, biasVal):
		Layer.__init__(self, layerName)
		self.layerType = 'Output'
		self.actFunction = actFunction
		self.weightInit = weightInit
		self.weightVal = float(weightVal)
		self.biasInit = biasInit
		self.biasVal = float(biasVal)