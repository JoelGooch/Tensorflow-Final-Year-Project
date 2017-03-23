
class Layer:
	def __init__(self, layerName):
		self.layerName = layerName

class ConvLayer(Layer):
	def __init__(self, layerName, kernelSize, stride, actFunction, numOutputFilters, padding, normalize, dropout , keepRate):
		Layer.__init__(self, layerName)
		self.layerType = 'Convolution'
		self.kernelSize = int(kernelSize)
		self.stride = int(stride)
		self.actFunction = actFunction
		self.numOutputFilters = int(numOutputFilters)
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
	def __init__(self, layerName, actFunction, numOutputNodes, normalize, dropout, keepRate):
		Layer.__init__(self, layerName)
		self.layerType = 'Dense'
		self.actFunction = actFunction
		self.numOutputNodes = int(numOutputNodes)
		self.normalize = normalize
		self.dropout = dropout
		self.keepRate = float(keepRate)

class OutputLayer(Layer):
	def __init__(self, layerName, actFunction):
		Layer.__init__(self, layerName)
		self.layerType = 'Output'
		self.actFunction = actFunction