import tensorflow as tf
import numpy as np

import whitebox as wb

from gen_data import gen_captcha, chars2ids, captcha_length, captcha_nchars

batch_size = 100
portable_saver_setting = {}

def get_batch():
	num_images_per_gen = 100
	_, image_arr = gen_captcha(num_images=1)
	height, width, nchannels = image_arr.shape
	
	def get_captcha():
		chars_batch, image_batch = zip(*gen_captcha(num_images=num_images_per_gen))
		chars_batch = map(lambda s: chars2ids(s), chars_batch)
		return np.array(chars_batch, np.int32), np.stack(image_batch)
	
	chars_batch, image_batch = tf.py_func(get_captcha, [], [tf.int32, tf.uint8])
	chars_batch = tf.reshape(chars_batch, [num_images_per_gen, captcha_length])
	image_batch = tf.reshape(image_batch, [num_images_per_gen, height, width, nchannels])
	captch_batch = tf.train.batch([chars_batch, image_batch], batch_size, num_threads=4, 
		capacity=batch_size*100, enqueue_many=True, name='captcha_batch')
	return captch_batch

def conv_layer(input_, out_channels, name='layer'):
	in_channels = input_.get_shape().as_list()[-1]
	weight = tf.get_variable(('%s_weight' % name), shape=[3, 3, in_channels, out_channels])
	bias = tf.get_variable(('%s_bias' % name), shape=[out_channels])
	portable_saver_setting['conv_%s_%s' % (name, weight)] = weight
	portable_saver_setting['conv_%s_%s' % (name, bias)] = bias
	conv_out = tf.nn.conv2d(input_, weight, strides=[1, 1, 1, 1], padding='SAME')
	add_out = tf.nn.bias_add(conv_out, bias)
	bn_out = tf.contrib.layers.batch_norm(add_out, center=True, scale=True, is_training=True)
	relu_out = tf.nn.relu(bn_out)
	wb.add(wb.WBKeys.FORWARD_NEURON_COVERAGE, relu_out, name='')
	pool_out = tf.nn.max_pool(relu_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return pool_out

def conv_section(input_, layers_setting):
	with tf.variable_scope('conv_section') and tf.name_scope('conv_section'):
		for i, layer in enumerate(layers_setting):
			output = conv_layer(input_, layer['width'], name=('layer%d' % i))
			input_ = output
	return output

def mlp_layer(input_, out_size, using_bn=False, activation=tf.nn.relu, name='layer'):
	in_size = input_.get_shape().as_list()[-1]
	weight = tf.get_variable(('%s_weight' % name), shape=[in_size, out_size])
	bias = tf.get_variable(('%s_bias' % name), shape=[out_size])
	portable_saver_setting['mlp_%s_%s' % (name, weight)] = weight
	portable_saver_setting['mlp_%s_%s' % (name, bias)] = bias
	li_out = tf.add(tf.matmul(input_, weight), bias)
	if using_bn:
		li_out = tf.contrib.layers.batch_norm(li_out, center=TRue, scale=True, is_training=True)
	acti_out = activation(li_out)
	if activation == tf.nn.relu:
		wb.add(wb.WBKeys.FORWARD_NEURON_COVERAGE, acti_out)
	return acti_out

def mlp_section(input_, setting):
	with tf.variable_scope('mlp_section') and tf.name_scope('mlp_section'):
		for i, layer in enumerate(setting):
			output = mlp_layer(input_, layer['width'], using_bn=layer['using_bn'],
				activation=layer['activation'], name=('layer%d' % i))
			input_ = output
	return output

def calc_loss(output, labels):
	output = tf.reshape(output, [-1, captcha_nchars])
	labels = tf.reshape(labels, [-1])
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)
	loss = tf.reduce_mean(loss)
	return loss

def create_model(input_, labels):
	conv_setting = [
		{'width': 32},
		{'width': 64},
		{'width': 64} ]
	mlp_setting = [
		{'width': 1024, 'using_bn': False, 'activation': tf.nn.relu},
		{'width': captcha_length * captcha_nchars, 'using_bn': False, 'activation': tf.identity} ]

	output = conv_section(input_, conv_setting)
	output = tf.reshape(output, [-1, np.prod(output.get_shape().as_list()[1:])])
	output = mlp_section(output, mlp_setting)
	
	output_loss = calc_loss(output, labels)
	
	output = tf.reshape(output, [-1, captcha_length, captcha_nchars])
	prediction = tf.cast(tf.argmax(output, 2), tf.int32)
	correct_chars = tf.equal(prediction, labels)
	correct_captcha = tf.reduce_all(correct_chars, axis=1)
	accuracy_chars = tf.reduce_mean(tf.cast(correct_chars, tf.float32))
	accuracy_captcha = tf.reduce_mean(tf.cast(correct_captcha, tf.float32))

	return {'output_loss': output_loss,
		'accuracy_chars': accuracy_chars,
		'accuracy_captcha': accuracy_captcha}


class Saver(object):
	def __init__(self):
		self._saver = tf.train.Saver()
		self._portable_saver = tf.train.Saver(portable_saver_setting)

	def restore(sess, path, mode=None):
		if mode == 'portable':
			self._portable_saver.restore(sess, path)
		else:
			self._saver.restore(sess, path)
	
	def save(sess, path, global_step=None, mode=None):
		if mode == 'portable':
			self._portable_saver.save(sess, path, global_step=global_step)
		else:
			self._saver.save(sess, path, global_step=global_step)
	
def train():
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			chars_batch, image_batch = get_batch()
			image_batch = tf.div(tf.cast(image_batch, tf.float32), 255)

		with tf.variable_scope('model') and tf.device('/gpu:0'):
			model = create_model(image_batch, chars_batch)
		
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(model['output_loss'])

		saver = Saver()
		wb_summary = wb.summary()

		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			try:
				step = 0
				step = 
				saver.restore(sess, './captcha_recognition.ckpt-%d' % step, mode='portable')
				while not coord.should_stop():
					step += 1
					sess.run(train_op)
					if step % 100 == 0:
						report = sess.run(model)
						print('loss: %f | acc_chars: %f | acc_captcha: %f | step: %d' % 
							(report['output_loss'], report['accuracy_chars'],
							 report['accuracy_captcha'], step))
						if report['accuracy_captcha'] > 0.95:
							saver.save(sess, '././captcha_recognition.ckpt', global_step=step)

			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
			except KeyboardInterrupt:
				print('KeyboardInterrupt raised')
			finally:
				coord.request_stop()
			coord.join(threads)

if __name__ == '__main__':
	train()