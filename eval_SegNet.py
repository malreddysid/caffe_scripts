# Siddarth Reddy Malreddy
# Script to run inference using the SegNet network.

import sys
sys.path.append('../../../python') # Need this path to import Caffe
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import time
import caffe
caffe.set_mode_cpu() # Use CPU for the calculations

print 'Reading the network and model'

# Download the .prototxt and .caffemodel from this URL - https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/Example_Models/segnet_model_driving_webdemo.prototxt and http://mi.eng.cam.ac.uk/%7Eagk34/resources/SegNet/segnet_weights_driving_webdemo.caffemodel

NET_FILE = 'segnet_model_driving_webdemo.prototxt'
MODEL_FILE = 'segnet_weights_driving_webdemo.caffemodel'
net = caffe.Net(NET_FILE, MODEL_FILE, caffe.TEST)

im_height = net.blobs['data'].data[0].shape[1]
im_width = net.blobs['data'].data[0].shape[2]

directory = '../../../../../car/' # Path to the image sequence
files = listdir(directory)
files.sort()

image_files = [files[i] for i in range(0, files.__len__(), 1) if files[i][-3:] == 'jpg']

print 'Starting processing'
total_time = 0 # To measure the average time taken
for file_name in image_files:
        print 'Processing image: ' + file_name
        image = cv2.imread(join(directory, file_name))
        image_shape = image.shape 
        image = cv2.resize(image, (im_width, im_height)) # Since SegNet has an encode and decoder style architecture it has some restrictions on the image size. So I rezied the input image to the size in which the training was done.
        image = image.transpose(2, 0, 1) # Preprocess the image to make it similar to the training data
        print image.shape


        net.blobs['data'].data[...] = image
        start = time.time()
        out = net.forward() # Run the network for the given input image
        end = time.time()
        print 'Time taken = ' + str(end - start)
        total_time += end - start

        out = out['argmax'] # Extract the output
        out = np.tile(np.squeeze(out), (1,1,1)) # Reshape the output into an image
        out = out.transpose(1, 2, 0)
        out = np.tile(out, (1,3))
        out = cv2.resize(out, (image_shape[1], image_shape[0]))
        out[out != 9] = 0 # 9 is the label 'car'
        out[out == 9] = 255

        out_file_name = file_name[:-4] + '_out' + '.png'
        cv2.imwrite(out_file_name, out)

print 'Done'
print 'Average time taken = ' + str(total_time/image_files.__len__())  
