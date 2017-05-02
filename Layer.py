
class Layer:
	def __init__(self, layer_name):
		self.layer_name = layer_name

class ConvLayer(Layer):
	def __init__(self, layer_name, kernel_size, stride, act_function, num_output_filters, weight_init, weight_val, bias_init, bias_val, padding, normalize, dropout , keep_rate):
		Layer.__init__(self, layer_name)
		self.layer_type = 'Convolution'
		self.kernel_size = int(kernel_size)
		self.stride = int(stride)
		self.act_function = act_function
		self.num_output_filters = int(num_output_filters)
		self.weight_init = weight_init
		self.weight_val = float(weight_val)
		self.bias_init = bias_init
		self.bias_val = float(bias_val)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keep_rate = float(keep_rate)

class MaxPoolingLayer(Layer):
	def __init__(self, layer_name, kernel_size, stride, padding, normalize, dropout, keep_rate):
		Layer.__init__(self, layer_name)
		self.layer_type = 'Max Pool'
		self.kernel_size = int(kernel_size)
		self.stride = int(stride)
		self.padding = padding
		self.normalize = normalize
		self.dropout = dropout
		self.keep_rate = float(keep_rate)

class FullyConnectedLayer(Layer):
	def __init__(self, layer_name, act_function, num_output_nodes, weight_init, weight_val, bias_init, bias_val, dropout, keep_rate):
		Layer.__init__(self, layer_name)
		self.layer_type = 'Fully Connected'
		self.act_function = act_function
		self.num_output_nodes = int(num_output_nodes)
		self.weight_init = weight_init
		self.weight_val = float(weight_val)
		self.bias_init = bias_init
		self.bias_val = float(bias_val)
		self.dropout = dropout
		self.keep_rate = float(keep_rate)

class OutputLayer(Layer):
	def __init__(self, layer_name, act_function, weight_init, weight_val, bias_init, bias_val):
		Layer.__init__(self, layer_name)
		self.layer_type = 'Output'
		self.act_function = act_function
		self.weight_init = weight_init
		self.weight_val = float(weight_val)
		self.bias_init = bias_init
		self.bias_val = float(bias_val)