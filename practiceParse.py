import xml.etree.ElementTree as ET


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


def getLayers(filePath):
	tree = ET.parse(filePath)
	root = tree.getroot()

	Layers = []

	for layer in root.iter('Layer'):
		layerType = layer.attrib['Type']

		for attribute in layer.getchildren():
			if attribute.tag == 'NumInputs':
				numInputs = attribute.text
			if attribute.tag == 'KernelSize':
				kernelSize = attribute.text
			if attribute.tag == 'Stride':
				stride = attribute.text
			if attribute.tag == 'ActFunction':
				actFunction = attribute.text
			if attribute.tag == 'NumOutputFilters':
				numOutputFilters = attribute.text
			if attribute.tag == 'Padding':
				padding = attribute.text
			if attribute.tag == 'Normalize':
				normalize = attribute.text
			if attribute.tag == 'Dropout':
				dropout = attribute.text
			if attribute.tag == 'KeepRate':
				keepRate = attribute.text
			if attribute.tag == 'NumOutputNodes':
				numOutputNodes = attribute.text


		if layerType == 'Convolution':
			layer = ConvLayer('Convolution', numInputs, kernelSize, stride, actFunction, numOutputFilters, padding, normalize, dropout, keepRate)
		if layerType == 'Max Pool':
			layer = MaxPoolingLayer('Max Pool', kernelSize, stride, padding, normalize, dropout, keepRate)
		if layerType == 'Dense':
			layer = DenseLayer('Dense', numInputs, actFunction, numOutputNodes, normalize, dropout, keepRate)
		if layerType == 'Output':
			layer = OutputLayer('Output', numInputs, actFunction, numOutputNodes)
		
		Layers.append(layer)

	return Layers


	'''
	for layer in Layers:
		attrs = vars(layer)
		print(', '.join("%s: %s" % item for item in attrs.items()))
	'''


