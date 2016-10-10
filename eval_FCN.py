# Siddarth Reddy Malreddy
# Script to run inference using the FCN network.

import sys
sys.path.append('../../../python') # Need this path to import Caffe
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from PIL import Image
import time
import caffe

caffe.set_mode_cpu() # Use CPU for the calculations

print 'Reading the network and model'

# Download the .prototxt and .caffemodel from this URL - https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s

NET_FILE = 'deploy.prototxt'
MODEL_FILE = 'fcn8s-heavy-pascal.caffemodel'
net = caffe.Net(NET_FILE, MODEL_FILE, caffe.TEST)

directory = '../../../../../car/' # Path to the image sequence
files = listdir(directory)
files.sort()

image_files = [files[i] for i in range(0, files.__len__(), 1) if files[i][-3:] == 'jpg'] #Get only image files

print 'Starting processing'
total_time = 0 # To measure the average time taken
for file_name in image_files:
        print 'Processing image: ' + file_name
        im = Image.open(join(directory, file_name))
        in_ = np.array(im, dtype=np.float32) # Preprocess the image to make it similar to the training data
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        net.blobs['data'].reshape(1, *in_.shape) # Reshape the network according to the image size
        net.blobs['data'].data[...] = in_
        start = time.time()
        net.forward() # Run the network for the given input image
        end = time.time()
        print 'Time taken = ' + str(end - start)
        total_time += end - start

        out = net.blobs['score'].data # Extract the output
        out = out.argmax(axis=1) # Get the labels at each pixel
        out = out.transpose(1, 2, 0) # Reshape the output into an image
        out = np.tile(out, (1,3))
        out[out != 7] = 0 # 7 is the label 'car'
        out[out == 7] = 255

        out_file_name = file_name[:-4] + '_out.png'
        cv2.imwrite(out_file_name, out)

print 'Done'
print 'Average time taken = ' + str(total_time/image_files.__len__())
