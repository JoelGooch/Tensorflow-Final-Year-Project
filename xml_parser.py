import xml.etree.ElementTree as ET
import Layer as l


def getLayers(filePath):
	tree = ET.parse(filePath)
	root = tree.getroot()

	Layers = []

	for layer in root.iter('Layer'):
		layerType = layer.attrib['Type']

		for attribute in layer.getchildren():
			if attribute.tag == 'LayerName':
				layerName = attribute.text
			if attribute.tag == 'KernelSize':
				kernelSize = attribute.text
			if attribute.tag == 'Stride':
				stride = attribute.text
			if attribute.tag == 'ActFunction':
				actFunction = attribute.text
			if attribute.tag == 'NumOutputFilters':
				numOutputFilters = attribute.text
			if attribute.tag == 'NumOutputNodes':
				numOutputNodes = attribute.text
			if attribute.tag == 'WeightInit':
				weightInit = attribute.text
			if attribute.tag == 'WeightVal':
				weightVal = attribute.text
			if attribute.tag == 'BiasInit':
				biasInit = attribute.text
			if attribute.tag == 'BiasVal':
				biasVal = attribute.text
			if attribute.tag == 'Padding':
				padding = attribute.text
			if attribute.tag == 'Normalize':
				normalize = attribute.text
			if attribute.tag == 'Dropout':
				dropout = attribute.text
			if attribute.tag == 'KeepRate':
				keepRate = attribute.text


		if layerType == 'Convolution':
			layer = l.ConvLayer(layerName, kernelSize, stride, actFunction, numOutputFilters, weightInit, weightVal, biasInit, biasVal, padding, normalize, dropout, keepRate)
		if layerType == 'Max Pool':
			layer = l.MaxPoolingLayer(layerName, kernelSize, stride, padding, normalize, dropout, keepRate)
		if layerType == 'Dense':
			layer = l.DenseLayer(layerName, actFunction, numOutputNodes, weightInit, weightVal, biasInit, biasVal, normalize, dropout, keepRate)
		if layerType == 'Output':
			layer = l.OutputLayer(layerName, actFunction, weightInit, weightVal, biasInit, biasVal,)
		
		Layers.append(layer)

	return Layers

def createXMLModel(layers, fileName, filePath):

	try:
		model = ET.Element('Model')

		for e in layers:
			layer = ET.SubElement(model, 'Layer', Type=e.layerType)
			if e.layerType == 'Convolution':
				layerName = ET.SubElement(layer, 'LayerName').text = e.layerName
				kernelSize = ET.SubElement(layer, 'KernelSize').text = str(e.kernelSize)
				stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
				actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
				numOutputFilters = ET.SubElement(layer, 'NumOutputFilters').text = str(e.numOutputFilters)
				weightInit = ET.SubElement(layer, 'WeightInit').text = str(e.weightInit)
				weightVal = ET.SubElement(layer, 'WeightVal').text = str(e.weightVal)
				biasInit = ET.SubElement(layer, 'BiasInit').text = str(e.biasInit)
				biasVal = ET.SubElement(layer, 'BiasVal').text = str(e.biasVal)
				padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
				normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
			elif e.layerType == 'Max Pool':
				layerName = ET.SubElement(layer, 'LayerName').text = e.layerName
				kernelSize = ET.SubElement(layer, 'KernelSize').text = str(e.kernelSize)
				stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
				padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
				normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
			elif e.layerType == 'Dense':
				layerName = ET.SubElement(layer, 'LayerName').text = e.layerName
				actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
				numOutputNodes = ET.SubElement(layer, 'NumOutputNodes').text = str(e.numOutputNodes)
				weightInit = ET.SubElement(layer, 'WeightInit').text = str(e.weightInit)
				weightVal = ET.SubElement(layer, 'WeightVal').text = str(e.weightVal)
				biasInit = ET.SubElement(layer, 'BiasInit').text = str(e.biasInit)
				biasVal = ET.SubElement(layer, 'BiasVal').text = str(e.biasVal)
				normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keepRate = ET.SubElement(layer, 'KeepRate').text = str(e.keepRate)
			elif e.layerType == 'Output':
				layerName = ET.SubElement(layer, 'LayerName').text = e.layerName
				actFunction = ET.SubElement(layer, 'ActFunction').text = e.actFunction
				weightInit = ET.SubElement(layer, 'WeightInit').text = str(e.weightInit)
				weightVal = ET.SubElement(layer, 'WeightVal').text = str(e.weightVal)
				biasInit = ET.SubElement(layer, 'BiasInit').text = str(e.biasInit)
				biasVal = ET.SubElement(layer, 'BiasVal').text = str(e.biasVal)

		tree = ET.ElementTree(model)

		fullPath = filePath + '/' + fileName
		tree.write(fullPath + ".xml")
		return True
	except:
		return False