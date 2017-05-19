import xml.etree.ElementTree as ET
import Layer as l


# this parses the .xml and returns all layers in the Layer class format
# @param file_path = string file directory where to load .xml
def get_layers(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()

	layers = []

	# cycle all elements in .xml
	for layer in root.iter('Layer'):
		layer_type = layer.attrib['Type']

		# branch depending on attribute found
		for attribute in layer.getchildren():
			if attribute.tag == 'LayerName':
				layer_name = attribute.text
			if attribute.tag == 'KernelSize':
				kernel_size = attribute.text
			if attribute.tag == 'Stride':
				stride = attribute.text
			if attribute.tag == 'ActFunction':
				act_function = attribute.text
			if attribute.tag == 'NumOutputFilters':
				num_output_filters = attribute.text
			if attribute.tag == 'NumOutputNodes':
				num_output_nodes = attribute.text
			if attribute.tag == 'WeightInit':
				weight_init = attribute.text
			if attribute.tag == 'WeightVal':
				weight_val = attribute.text
			if attribute.tag == 'BiasInit':
				bias_init = attribute.text
			if attribute.tag == 'BiasVal':
				bias_val = attribute.text
			if attribute.tag == 'Padding':
				padding = attribute.text
			if attribute.tag == 'Normalize':
				normalize = attribute.text
			if attribute.tag == 'Dropout':
				dropout = attribute.text
			if attribute.tag == 'KeepRate':
				keep_rate = attribute.text

		# create Layer and add to list of layers
		if layer_type == 'Convolution':
			layer = l.ConvLayer(layer_name, kernel_size, stride, act_function, num_output_filters, weight_init, weight_val, bias_init, bias_val, padding, normalize, dropout, keep_rate)
		if layer_type == 'Max Pool':
			layer = l.MaxPoolingLayer(layer_name, kernel_size, stride, padding, normalize, dropout, keep_rate)
		if layer_type == 'Fully Connected':
			layer = l.FullyConnectedLayer(layer_name, act_function, num_output_nodes, weight_init, weight_val, bias_init, bias_val, dropout, keep_rate)
		if layer_type == 'Output':
			layer = l.OutputLayer(layer_name, act_function, weight_init, weight_val, bias_init, bias_val,)
		
		layers.append(layer)

	return layers


# this parses the the layers in Layer class format into an .xml file
# @param file_name = string of name to save new .xml file as
# @param file_path = string file directory where to save .xml
def create_XML_model(layers, file_name, file_path):

	try:
		model = ET.Element('Model')

		# branch depending on which type the current layer is
		for e in layers:
			layer = ET.SubElement(model, 'Layer', Type=e.layer_type)
			if e.layer_type == 'Convolution':
				layer_name = ET.SubElement(layer, 'LayerName').text = e.layer_name
				kernel_size = ET.SubElement(layer, 'KernelSize').text = str(e.kernel_size)
				stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
				act_function = ET.SubElement(layer, 'ActFunction').text = e.act_function
				num_output_filters = ET.SubElement(layer, 'NumOutputFilters').text = str(e.num_output_filters)
				weight_init = ET.SubElement(layer, 'WeightInit').text = e.weight_init
				weight_val = ET.SubElement(layer, 'WeightVal').text = str(e.weight_val)
				bias_init = ET.SubElement(layer, 'BiasInit').text = e.bias_init
				bias_val = ET.SubElement(layer, 'BiasVal').text = str(e.bias_val)
				padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
				normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keep_rate = ET.SubElement(layer, 'KeepRate').text = str(e.keep_rate)
			elif e.layer_type == 'Max Pool':
				layer_name = ET.SubElement(layer, 'LayerName').text = e.layer_name
				kernel_size = ET.SubElement(layer, 'KernelSize').text = str(e.kernel_size)
				stride = ET.SubElement(layer, 'Stride').text = str(e.stride)
				padding = ET.SubElement(layer, 'Padding').text = str(e.padding)
				normalize = ET.SubElement(layer, 'Normalize').text = str(e.normalize)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keep_rate = ET.SubElement(layer, 'KeepRate').text = str(e.keep_rate)
			elif e.layer_type == 'Fully Connected':
				layer_name = ET.SubElement(layer, 'LayerName').text = e.layer_name
				act_function = ET.SubElement(layer, 'ActFunction').text = e.act_function
				num_output_nodes = ET.SubElement(layer, 'NumOutputNodes').text = str(e.num_output_nodes)
				weight_init = ET.SubElement(layer, 'WeightInit').text = e.weight_init
				weight_val = ET.SubElement(layer, 'WeightVal').text = str(e.weight_val)
				bias_init = ET.SubElement(layer, 'BiasInit').text = e.bias_init
				bias_val = ET.SubElement(layer, 'BiasVal').text = str(e.bias_val)
				dropout = ET.SubElement(layer, 'Dropout').text = str(e.dropout)
				keep_rate = ET.SubElement(layer, 'KeepRate').text = str(e.keep_rate)
			elif e.layer_type == 'Output':
				layer_name = ET.SubElement(layer, 'LayerName').text = e.layer_name
				act_function = ET.SubElement(layer, 'ActFunction').text = e.act_function
				weight_init = ET.SubElement(layer, 'WeightInit').text = e.weight_init
				weight_val = ET.SubElement(layer, 'WeightVal').text = str(e.weight_val)
				bias_init = ET.SubElement(layer, 'BiasInit').text = e.bias_init
				bias_val = ET.SubElement(layer, 'BiasVal').text = str(e.bias_val)

		tree = ET.ElementTree(model)

		full_path = file_path + '/' + file_name
		tree.write(full_path + ".xml")
		return True
	except:
		return False