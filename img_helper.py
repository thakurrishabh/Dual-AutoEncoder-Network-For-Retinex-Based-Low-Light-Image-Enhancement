#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from random import uniform
from skimage.color import rgb2hsv,hsv2rgb
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img


# Coverting images from RGB to HSV colour space
def rgb_to_hsv_images(image_array):

	image_hsv_arr = np.ndarray(shape=image_array.shape, dtype=np.float32)

	c = 0
	for image_in_rgb in image_array:
		image_in_hsv = rgb2hsv(image_in_rgb)
		image_hsv_arr[c] = image_in_hsv
		c = c+1

	return image_hsv_arr

# Coverting images from HSV to RGB colour space
def hsv_to_rgb_images(image_array):

	image_rgb_arr = np.ndarray(shape=image_array.shape, dtype=np.float32)

	c = 0
	for image_in_hsv in image_array:
		image_in_rgb = hsv2rgb(image_in_hsv)
		image_rgb_arr[c] = image_in_rgb
		c = c+1

	return image_rgb_arr

# Illuminate a set of HSV images using gamma correction
def image_illumination_C(image_array,image_width):

	image_height = image_width
	no_of_channels = 3
	aray_of_illuminated_images = np.ndarray(shape=image_array.shape, dtype=np.float32)

	c = 0
	for v in image_array:
		image =  v.copy()
		illumi= 1 / uniform(1.6, 3.3)
		# Apply gamma correction to enhance illumination
		for a in range(image_width):
			for b in range(image_height):
				image[a,b,2] = pow(image[a,b,2], illumi)

		aray_of_illuminated_images[c] = image
		c = c+1

	return aray_of_illuminated_images

def convert_To_Low_Light_images(image_array,image_width):

	image_height = image_width
	no_of_channels = 3
	array_of_Low_Light_images = np.ndarray(shape=image_array.shape, dtype=np.float32)

	c = 0
	for v in image_array:
		image =  v.copy()
		illumi= 3.3
		# Apply gamma correction to enhance illumination
		for a in range(image_width):
			for b in range(image_height):
				image[a,b,2] = pow(image[a,b,2], illumi)

		array_of_Low_Light_images[c] = image
		c = c+1

	return array_of_Low_Light_images

# To add random noise to Value channel on a array of HSV images
def noise_generator(image_array,image_width):

	image_height = image_width
	no_of_channels = 3
	noisy_images = np.ndarray(shape=image_array.shape, dtype=np.float32)

	# Create random Gaussian distributed noise
	noise_level = 0.01
	mean_n = 0.0
	stand_deviation_n = uniform(10, 18)

	n_value =  np.random.normal(loc=mean_n,
				scale=stand_deviation_n,
				size=(len(image_array), image_width, image_height))

	for i in range(len(image_array)):
		noisy_images[i,:,:,0] = image_array[i,:,:,0]
		noisy_images[i,:,:,1] = image_array[i,:,:,1]
		noisy_images[i,:,:,2] = image_array[i,:,:,2].copy() + (n_value[i,:,:] * noise_level)

	# Keep pixel values in range [0,1]
	noisy_images = np.clip(noisy_images, 0., 1.)

	return noisy_images

# Reduce a set of images to 1 colour channel
#reduce to 1
def channel_extractor_c1(image_array,image_width,cnl):
 
	image_height = image_width
	no_of_channels = 3
	extracted_channel = np.ndarray(shape=(len(image_array), image_width, image_height, 1),
	                    	dtype=np.float32)

	c = 0
	for image in image_array:
		extracted_channel[c,:,:,0] = image[:,:,cnl]
		c = c+1

	return extracted_channel

# Update the value channel in a set of HSV images
#To update the value channel in the HSV images
def v_channel_upd(image_array, V_set):

	upd_channel = np.ndarray(shape=image_array.shape, dtype=np.float32)

	for i in range(len(image_array)):
		upd_channel[i,:,:,0] = image_array[i,:,:,0]
		upd_channel[i,:,:,1] = image_array[i,:,:,1]
		upd_channel[i,:,:,2] = V_set[i,:,:]

	return upd_channel

# Reduce a set of HSV images to just the value channel
#reduce to v
def channel_reducer_cv(image_array):

	channel_reducer = np.ndarray(shape=image_array.shape, dtype=np.float32)

	c = 0
	for image in image_array:
		channel_reducer[c,:,:,0] = 0
		channel_reducer[c,:,:,1] = 0
		channel_reducer[c,:,:,2] = image[:,:,2]
		c = c+1

	return channel_reducer

# Create a plot of images
def plotting_func(row, col, d, t, size):

	figsize = [size, size]
	fig, ax = plt.subplots(nrows=row, ncols=col, figsize=figsize)

	for j in range(row):
		arr = np.asarray(d[j])
		ax[j,2].set_title(t[j])

		for i in range(col):
			image = array_to_img(arr[i])
			ax[j,i].imshow(image)
			ax[j,i].axis('off')

	return fig

