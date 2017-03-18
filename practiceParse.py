import xml.etree.ElementTree as ET
import Layer as l


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
			layer = l.ConvLayer('Convolution', numInputs, kernelSize, stride, actFunction, numOutputFilters, padding, normalize, dropout, keepRate)
		if layerType == 'Max Pool':
			layer = l.MaxPoolingLayer('Max Pool', kernelSize, stride, padding, normalize, dropout, keepRate)
		if layerType == 'Dense':
			layer = l.DenseLayer('Dense', numInputs, actFunction, numOutputNodes, normalize, dropout, keepRate)
		if layerType == 'Output':
			layer = l.OutputLayer('Output', numInputs, actFunction, numOutputNodes)
		
		Layers.append(layer)

	return Layers

def createXMLModel(layers):
	model = ET.Element('Model')

	for e in layers:
		layer = ET.SubElement(model, 'Layer', Type=e.layerType)
		if e.layerType == 'Convolution':
			numInputs = ET.SubElement(layer, 'NumInputs').text = str(e.numInputs)
			kernelSize = ET.SubElement(layer, 'KernelSize').text = str(e.kernelSize)
			stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
			actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
			numOutputFilters = ET.SubElement(layer, 'NumOutputFilters').text = str(e.numOutputFilters)
			padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
			normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
			dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
			keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
		elif e.layerType == 'Max Pool':
			kernelSize = ET.SubElement(layer, 'KernelSize').text = str(e.kernelSize)
			stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
			padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
			normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
			dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
			keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
		elif e.layerType == 'Dense':
			numInputs = ET.SubElement(layer, 'NumInputs').text = str(e.numInputs)
			actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
			numOutputNodes = ET.SubElement(layer, 'NumOutputNodes').text = str(e.numOutputNodes)
			normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
			dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
			keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
		elif e.layerType == 'Output':
			numInputs = ET.SubElement(layer, 'NumInputs').text = str(e.numInputs)
			actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
			numOutputNodes = ET.SubElement(layer, 'NumOutputNodes').text = str(e.numOutputNodes)

	tree = ET.ElementTree(model)
	complete = tree.write("TestModel.xml")
	return complete