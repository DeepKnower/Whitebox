import os
import random

from captcha.image import ImageCaptcha
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf



number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
			'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
			'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
			'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
			'U', 'V', 'W', 'X', 'Y', 'Z']

basepath_data = "/home/jarvis2/datasets/captcha_recognition/data"
basepath_images = "/home/jarvis2/datasets/captcha_recognition/images"

num_images_for_train = 0 #10000000
num_images_for_valid = 10000
num_images_per_datafile = 10000

captcha_height=60
captcha_width=160
captcha_length=4
captcha_mode='naA'
captcha_nchars=10 + 26 + 26 + 1 # 1 for unknown

def char2id(c):
	code = ord(c)
	if code >= 48 and code <= 57:
		return code - 48
	elif code >=65 and code <= 90:
		return code - 65 + 10
	elif code >=97 and code <=122:
		return code - 97 + 10 + 26
	else:
		raise ValueError('Invalid character')

def chars2ids(cs):
	return [char2id(c) for c in cs]

def gen_characters(len_chars=4, mode='naA'):
	all_chars = []
	if 'n' in mode:
		all_chars += number
	if 'a' in mode:
		all_chars += alphabet
	if 'A' in mode:
		all_chars += ALPHABET
	characters = [random.choice(all_chars) for i in range(len_chars)]
	return ''.join(characters)

def _gen_captcha(ic, characters):
	while True:
		image_io = Image.open(ic.generate(characters))
		image_arr = np.array(image_io)
		if image_arr.shape == (captcha_height, captcha_width, 3):
			return image_arr

def gen_captcha(num_images=1, characters=None, len_chars=4, mode='naA'):
	if characters:
		if isinstance(characters, (list, tuple)):
			num_images = len(characters)
		elif isinstance(characters, str):
			num_images = 1
		else:
			raise Exception('`characters` is invalid')

	ic = ImageCaptcha(width=captcha_width, height=captcha_height)
	if num_images == 1:
		if not characters:
			characters = gen_characters(len_chars, mode)
			image_arr = _gen_captcha(ic, characters)
		return (characters, image_arr)
	elif num_images > 1:
		if not characters:
			characters = [gen_characters(len_chars, mode) for i in range(num_images)]
		return [(chars, _gen_captcha(ic, chars)) for chars in characters]
	else:
		raise Exception('`num_images` is invalid')

def save_captcha(data, filepath, output_image=False, image_basepath=None, image_prefix='image'):
	with tf.python_io.TFRecordWriter(filepath) as writer:
		for i, (chars, image_arr) in enumerate(data):
			height = image_arr.shape[0]
			width = image_arr.shape[1]
			image_bytes = image_arr.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
				'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
				'image_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
				}))
			writer.write(example.SerializeToString())
			if output_image and image_basepath:
				image_filename = '%s_%d_%s.png' % (image_prefix, i, chars)
				Image.fromarray(image_arr).save(os.path.join(image_basepath, image_filename), 'PNG')

def gen_data():
	if not os.path.exists(basepath_data):
		os.mkdir(basepath_data)
	if not os.path.exists(basepath_images):
		os.mkdir(basepath_images)
	
	num_datafiles = int(np.ceil(num_images_for_train / num_images_per_datafile))
	for i in range(num_datafiles):
		captcha_data = gen_captcha(
			num_images=num_images_per_datafile,
			characters=None,
			len_chars=captcha_length,
			mode=captcha_mode)
		save_captcha(captcha_data, os.path.join(basepath_data, ('train_%d.tfr' % i)))

	captcha_data = gen_captcha(
		num_images=num_images_for_valid,
		characters=None,
		len_chars=captcha_length,
		mode=captcha_mode)
	save_captcha(captcha_data, os.path.join(basepath_data, 'valid.tfr'), output_image=True,
		image_basepath=basepath_images, image_prefix='valid')

if __name__ == '__main__':
	gen_data()

