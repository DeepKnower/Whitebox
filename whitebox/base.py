import tensorflow as tf

class Keys(object):
	FORWARD_NEURON_COVERAGE = "wb_forward_neuron_coverage"
	BACKWARD_NEURON_COVERAGE = "wb_backward_neuron_coverage"
	#NEURON_GRADIENT_SCALE = "wb_backw"

class Collections(object):
	def __init__(self):
		self.wb_forward_neuron_coverage = {}
		self.wb_backward_neuron_coverage = {}

	def get(self, key):
		if not hasattr(self, key):
			raise ValueError('Invalid WBKeys: %s' % key)
		else:
			return getattr(self, key)

_default_collections = Collections()

class Summary(object):
	def __init__(self):
		self.summary_list = []
		self.loss = None

	def compute_activation_ratio(self, tensor):
		return tf.reduce_mean(tf.cast(tf.greater(tensor, 0.), tf.float32))

	def compute_conduction_ratio(self, tensor):
		return tf.reduce_mean(tf.cast(tf.greater(tf.abs(tensor), 1e-6), tf.float32))

	def _summarize_per_batch(self, key, fn):
		graph = tf.get_default_graph()
		for name in graph.get_collection(key):
			tensor = _default_collections.get(key)[name]
			indicator = fn(tensor)
			self.summary_list.append(tf.summary.scalar(('%s/%s' % (key, name)), indicator))

	def _summarize_gradient_per_batch(self, key, fn):
		graph = tf.get_default_graph()
		xs = []
		for name in graph.get_collection(key):
			tensor = _default_collections.get(key)[name]
			xs.append(tensor)
		grads = tf.gradients(self.loss, xs)
		for g in grads:
			indicator = fn(g)
			self.summary_list.append(tf.summary.scalar(('%s/%s' % (key, name)), indicator))			

	def summarize(self):
		self._summarize_per_batch(Keys.FORWARD_NEURON_COVERAGE, self.compute_activation_ratio)
		self._summarize_gradient_per_batch(Keys.BACKWARD_NEURON_COVERAGE, self.compute_conduction_ratio)

_default_summary = Summary()

def add(key, tensor, name=None):
	graph = tf.get_default_graph()
	nscope = graph._name_stack
	if not name:
		name = tensor.op.name.split('/')[-1]
	if nscope:
		name = '%s/%s' % (nscope, name)
	_default_collections.get(key)[name] = tensor
	graph.add_to_collection(key, name)

def summary(loss=None):
	if loss is not None:
		_default_summary.loss = loss
	_default_summary.summarize()
	return tf.summary.merge(_default_summary.summary_list)