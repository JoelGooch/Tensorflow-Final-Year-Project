
class Layer:
	def __init__(self, layerType):
		self.layerType = layerType

class ConvLayer(Layer):
	def __init__(self, layerType, numInputs, kernelSize, stride, actFunction, numOutputFilters, padding, normalize, dropout , keepRate):
		Layer.__init__(self, layerType)
		self.numInputs = int(numInputs)
		self.kernelSize = int(kernelSize)
		self.stride = int(stride)
		self.actFunction = actFunction
		self.numOutputFilters = int(numOutputFilters)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class MaxPoolingLayer(Layer):
	def __init__(self, layerType, kernelSize, stride, padding, normalize, dropout, keepRate):
		Layer.__init__(self, layerType)
		self.kernelSize = int(kernelSize)
		self.stride = int(stride)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class DenseLayer(Layer):
	def __init__(self, layerType, numInputs, actFunction, numOutputNodes, normalize, dropout, keepRate):
		Layer.__init__(self, layerType)
		self.numInputs = int(numInputs)
		self.actFunction = actFunction
		self.numOutputNodes = int(numOutputNodes)
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class OutputLayer(Layer):
	def __init__(self, layerType, numInputs, actFunction, numOutputNodes):
		Layer.__init__(self, layerType)
		self.numInputs = int(numInputs)
		self.actFunction = actFunction
		self.numOutputNodes = int(numOutputNodes)