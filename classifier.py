#!/usr/bin/env python

# Based off http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

import caffe
from caffe.proto import caffe_pb2
from ConfigParser import RawConfigParser
import cv2
import lmdb
import numpy as np
from pprint import pprint
import praw
import os
from requests import get
from shutil import copyfileobj
from sys import argv, exit
from time import sleep

USER_AGENT = 'Dog/Cat Classification 1.0 by [handle]'
SUBREDDIT = 'dogpictures+catpictures'

# Size to process to.
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def get_images_to_classify(user, pw, img_dir):
    """Login to reddit and download images and necessary info to classify."""
    r = praw.Reddit(user_agent=USER_AGENT)
    r.login(username=user, password=pw, disable_warning=True)
    subreddit = r.get_subreddit(SUBREDDIT)
    file_tups = []
    for i, submission in enumerate(subreddit.get_hot(limit=20)):
        sub_id = submission.id
        sub_name = submission.subreddit.display_name
        # API provides multiple resolutions for each submission's image.
        resolutions = submission.preview['images'][0]['resolutions']
        num_res = len(resolutions)
        # Get the middle resolution's index (if even, round down).
        mid_index = num_res / 2 - 1 if num_res % 2 == 0 else num_res / 2
        img_url = resolutions[mid_index]['url']
        response = get(img_url, stream=True)
        fname = img_dir + '/' + 'img' + str(i) + '_' + sub_name + '_' + sub_id + '.jpg'
	# This avoids reading the whole file into memory at once.
        with open(fname, 'wb') as out_file:
            copyfileobj(response.raw, out_file)
	    file_tups.append((fname, sub_id))
        del response
    return file_tups

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    """Image processing helper function."""
    # Histogram equalization.
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image resizing.
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def predict_image_class(file_tups):
    """For each of the downloaded images, try to classify it as a dog or cat."""
    caffe.set_mode_gpu()

    # Reading mean image, caffe model, and its weights.

    # Read mean image.
    mean_blob = caffe_pb2.BlobProto()
    with open('/home/ubuntu/deeplearning-cats-dogs-tutorial/input/mean.binaryproto') as f:
    	mean_blob.ParseFromString(f.read())
	mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))

    # Read model architecture and trained model's weights.
    net = caffe.Net('/home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt',
                '/home/ubuntu/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1_iter_5000.caffemodel',
                caffe.TEST)

    # Define image transformers.
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2,0,1))

    # Get list of filepaths.
    test_img_paths = [tup[0] for tup in file_tups]

    # Making predictions.
    test_ids = []
    preds = []
    pred_tuples = []
    for i, img_path in enumerate(test_img_paths):
    	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    	img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    	net.blobs['data'].data[...] = transformer.preprocess('data', img)
    	out = net.forward()
    	pred_probas = out['prob']

    	test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    	preds = preds + [pred_probas.argmax()]

    	print img_path
	# The class with the higher probability: 1 = dog; 0 = cat.
    	print pred_probas.argmax()
	print pred_probas
    	print "-------"
	# Tuple of the filepath, submission id, and its predicted class.
	pred_tuples.append((img_path, file_tups[i][1], pred_probas.argmax()))

    # Write all predictions to output file.
    with open('../caffe_models/caffe_model_1/submission_model_1.csv','w') as f:
        f.write("id,label\n")
    	for i in range(len(test_ids)):
        	f.write(str(test_ids[i])+","+str(preds[i])+"\n")

    return pred_tuples

def reply_with_class(user, pw, pred_tuples):
    """Reply to submissions with a representative picture of the predicted class."""
    r = praw.Reddit(user_agent=USER_AGENT)
    r.login(username=user, password=pw, disable_warning=True)
    for tup in pred_tuples:
        # Check that the prediction matches to the subreddit in the filename.
        fname, sub_id, pred = tup[0], tup[1], tup[2]
        # If prediction wrong, don't reply.
        if ('dogpicture' in fname and pred == 0) or ('catpicture' in fname and pred == 1): continue
        submission = r.get_submission(submission_id = sub_id)
        if pred == 1:
	    submission.add_comment("http://cdn2-www.dogtime.com/assets/uploads/gallery/30-impossibly-cute-puppies/impossibly-cute-puppy-8.jpg")
        elif pred == 0:
	    submission.add_comment("http://www.welikeviral.com/files/2014/12/8325_8_site_clear.jpeg")
	# To not hit rate limit.
	sleep(30)

def main():
    try:
        config_filepath = argv[1]
    except IndexError as err:
        print "ERROR: " + str(err)
        print "Please supply a config file."
        exit()
    config = RawConfigParser()
    config.read(config_filepath)
    username = config.get('Main', 'username')
    password = config.get('Main', 'password')
    image_dir = config.get('Main', 'image_dir')
    try:
        file_tups = get_images_to_classify(username, password, image_dir)
        pred_tuples = predict_image_class(file_tups)
        reply_with_class(username, password, pred_tuples)
    except praw.errors.APIException as e:
        print "ERROR: ", e
    # Shouldn't happen since PRAW controls the rate automatically, but just in case.
    except praw.errors.RateLimitExceeded as e:
        print "ERROR: Rate Limit Exceeded: {}".format(e)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [502, 503, 504]:
            # May be temporary.
	    pass
        else:
            # Assume others are fatal.
            print "ERROR: {}".format(e)
            print "Fatal, exiting"
    except Exception as e:
        print "ERROR: ", e
        print "Lazily handling error, exiting."
        exit()

if __name__ == '__main__':
    main()
