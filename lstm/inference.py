import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op, dtypes
import numpy as np
import sys, os, argparse, configparser, math, cv2, pickle, json

def loadINI(iniPath, i):
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(iniPath)
	return config[i]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', default='config.ini', help='Path to config file.')
	parser.add_argument('-i', '--index', default='0', help='[i] in config file.')
	parser.add_argument('-m', '--model', default='m0', help='Model identifier.')
	parser.add_argument('-v', '--videoListPath', dest='videoListPath', default='vids.txt', help='Path to video list file.')
	parser.add_argument('-g', '--gpu', default=0, help='GPU number to use.')
	return parser.parse_args()

def getNormalizedTravelInfo(bbDict):
	length = len(bbDict)
	posDict = {}
	x0_a,y0_a = 0,0
	for f in bbDict.keys():
		x1,y1,w,h = bbDict[f]
		x0,y0 = x1+0.5*w, y1+0.5*h
		posDict[f] = (x0,y0)
		x0_a,y0_a = x0+x0_a,y0+y0_a
	x0_a,y0_a = x0_a/length,y0_a/length
	travelIndex = 0
	for f in posDict.keys():
		x0,y0 = posDict[f]
		travelIndex += (abs(x0-x0_a)+abs(y0-y0_a))/length
	if travelIndex<=1:
		travelIndex = 0
	else:
		travelIndex = math.pow(1-1/travelIndex, 2)
	bx,ex,by,ey = 0,0,0,0
	beginFrame, endFrame = min(bbDict.keys()), max(bbDict.keys())
	p = int(length/4)
	bxp,byp = posDict[beginFrame]
	exp,eyp = posDict[endFrame-p]
	for i in range(1, p):
		bxc,byc = posDict[beginFrame+i]
		exc,eyc = posDict[endFrame-p+i]
		bx,ex,by,ey = bx+(bxc-bxp)/p,ex+(exc-exp)/p,by+(byc-byp)/p,ey+(eyc-eyp)/p
		bxp,exp,byp,eyp = bxc,exc,byc,eyc
	travelAngle = (math.sin(math.atan2(bx*ey-by*ex,bx*ex+by*ey))+1)/2
	return [[travelIndex, travelAngle]]

def getNormalizedBoundingBoxInfo(bbDict, original_size):
	original_width, original_height = original_size
	bbInfo = []
	for f in bbDict.keys():
		x1,y1,w,h = bbDict[f]
		x0,y0 = x1+0.5*w, y1+0.5*h
		bbInfo.append([x0/original_width, y0/original_height, w/original_width, h/original_height])
	return linSample(bbInfo)

def readImages(videoFilePath, trkDictFilePath, input_size):
	with open(trkDictFilePath, 'rb') as trkDictFile:
		trkDict = pickle.load(trkDictFile)
	objDict = {}
	cap = cv2.VideoCapture(videoFilePath)
	original_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	try:
		ok, image = cap.read()
	except:
		print('Error reading video '+videoFilePath+'.')
		return {}
	f = 1
	while ok:
		sys.stdout.write('\rRead: '+str(f))
		sys.stdout.flush()
		for trkID in trkDict.keys():
			bbDict, objType, _, _, _ = trkDict[trkID]
			if f not in bbDict.keys():
				continue
			x1,y1,w,h = bbDict[f]
			if trkID not in objDict.keys():
				objDict[trkID] = objType, bbDict, original_size, {}
			_,_,_,imgDict = objDict[trkID]
			imgDict[f] = cv2.resize(cv2.getRectSubPix(image, (w,h), (int(x1+0.5*w),int(y1+0.5*h))), input_size)
		ok, image = cap.read()
		f += 1
	sys.stdout.write('\n')
	return objDict

def linSample(array):
	return np.array(array)[np.linspace(0, len(array), steps, endpoint=False, dtype=np.int32).tolist()]

def form_feedDict(imgDict, original_size, fRange, bbDict):
	fd_label = np.reshape([0], (-1, 1))
	imgList = []
	for f in linSample(sorted([f for f in imgDict.keys() if f in fRange])):
		imgList.append(imgDict[f])
	fd_images = np.stack(imgList)
	fd_travel = getNormalizedTravelInfo(bbDict)
	fd_bbox = getNormalizedBoundingBoxInfo(bbDict, original_size)
	return {label: fd_label, images: fd_images, travel: fd_travel, bbox: fd_bbox, is_training: False}

def lstm_cell(W, b, forget_bias, inputs, state):
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = tf.nn.relu
	c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
	gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, b)
	i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
	forget_bias_tensor = constant_op.constant(forget_bias, dtype=f.dtype)
	new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), activation(j)))
	new_h = multiply(activation(new_c), sigmoid(o))
	new_state = array_ops.concat([new_c, new_h], 1)
	return new_h, new_state

def broadcast(tensor, shape):
	return tensor+tf.zeros(shape, dtype=tensor.dtype)

# Main
if __name__ == '__main__':

	# Load config and gets variables
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	videoListPath = args.videoListPath
	varDict = loadINI(args.config, args.index)
	input_width,input_height = int(varDict['input_width']),int(varDict['input_height'])
	num_channels = int(varDict['num_channels'])
	n_hidden1,n_hidden2 = int(varDict['n_hidden1']),int(varDict['n_hidden2'])
	feature_size = int(varDict['feature_size'])
	steps = int(varDict['steps'])
	dropout_keep_prob = float(varDict['dropout_keep_prob'])
	videosFolder = varDict['videosFolder']
	tracksFolder = varDict['tracksFolder']
	activitiesFolder = varDict['activitiesFolder']
	detector = varDict['detector']
	tracker = varDict['tracker']
	regressor = varDict['regressor']
	inDesc = detector+'+'+tracker
	outDesc = detector+'+'+tracker+'+'+regressor+'_'+args.model
	inFolder = os.path.join(tracksFolder, inDesc)
	outFolder = os.path.join(activitiesFolder, outDesc)
	chkptPath = os.path.join(regressor, 'chkpt', 'vgg16-'+args.model)
	windowSize = int(varDict['windowSize'])
	windowOverlap = float(varDict['windowOverlap'])
	bgLabelPerson = varDict['bgPerson']
	bgLabelCar = varDict['bgCar']
	activities = varDict['activities'].split(',')
	activities.append(bgLabelPerson)
	activities.append(bgLabelCar)
	classes = 2
	
	label = tf.placeholder(tf.int64, [None, 1])
	images = tf.placeholder(tf.float32, [None, input_width, input_height, num_channels])
	travel = tf.placeholder(tf.float32, [None, 2])
	bbox = tf.placeholder(tf.float32, [None, 4])
	is_training = tf.placeholder(tf.bool)
	learning_rate = tf.placeholder(tf.float32, [])
	
	# Define LSTM
	init_state1 = vs.get_variable('s1', shape=[2*n_hidden1])
	W_lstm1 = vs.get_variable('W1', shape=[feature_size+2*n_hidden1+2+4, 4*n_hidden1]) # +2 for travel and +4 for bbox
	b_lstm1 = vs.get_variable('b1', shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))
	curr_state1 = broadcast(init_state1, [1, 2*n_hidden1])
	_, out_lstm1 = array_ops.split(value=curr_state1, num_or_size_splits=2, axis=1)
	init_state2 = vs.get_variable('s2', shape=[2*n_hidden1])
	W_lstm2 = vs.get_variable('W2', shape=[2*n_hidden1, 4*n_hidden1])
	b_lstm2 = vs.get_variable('b2', shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))
	curr_state2 = broadcast(init_state2, [1, 2*n_hidden1])
	_, out_lstm2 = array_ops.split(value=curr_state2, num_or_size_splits=2, axis=1)
	
	b, g, r = tf.split(axis=3, num_or_size_splits=3, value=images*255.0)
	VGG_MEAN = [103.939, 116.779, 123.68]
	images = tf.concat(values=[b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], axis=3)

	with tf.variable_scope('vgg_16', [images]) as sc:
		end_points_collection = sc.original_name_scope+'_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		slim = tf.contrib.slim
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
			net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			net = slim.max_pool2d(net, [2, 2], scope='pool5')
			# Use conv2d instead of fully_connected layers.
			net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			vgg16_Features = tf.reshape(net, (-1, 4096))
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

	for i in range(steps):
		lstm_input = array_ops.concat([tf.reshape(vgg16_Features[i,:], (-1, feature_size)), out_lstm2, [travel[0]], [bbox[i,:]]], 1)
		h_1, curr_state1 = lstm_cell(W_lstm1, b_lstm1, 1.0, lstm_input, curr_state1)
		h_1 = tf.cond(is_training, lambda: tf.nn.dropout(h_1, keep_prob=dropout_keep_prob), lambda: h_1)
		out_lstm1 = h_1
		h_2, curr_state2 = lstm_cell(W_lstm2, b_lstm2, 1.0, out_lstm1, curr_state2)
		h_2 = tf.cond(is_training, lambda: tf.nn.dropout(h_2, keep_prob=dropout_keep_prob), lambda: h_2)
		out_lstm2 = h_2

	W_fc1 = tf.Variable(tf.truncated_normal([n_hidden1, classes], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[classes]))
	fc1 = tf.matmul(out_lstm2, W_fc1) + b_fc1

	# Loss function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(label, classes), logits=fc1))

	# Evaluation
	top_val, top_class = tf.nn.top_k(fc1, classes)
	result = tf.argmax(fc1, 1)
	top_softmax = tf.nn.softmax(top_val)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=0)

	with tf.Session() as sess:
		sess.run(init)

		# Create actDict
		with open(videoListPath, 'r') as vidListFile:
			videos = vidListFile.read().splitlines()
		for video in videos:
			actID = 1000000*int('1'+video.replace('VIRAT_S_','').replace('_',''))
			csvList = []
			actDict = {}
			objDict = readImages(os.path.join(videosFolder, video+'.mp4'), os.path.join(inFolder, video+'.pickle'), (input_width,input_height))
			for activity in activities:
				chkpts = [chkptFile.replace('.index','') for chkptFile in os.listdir(chkptPath) if chkptFile.startswith(activity+'-') and chkptFile.endswith('.index')]
				if len(chkpts)==0:
					continue
				chkpt = os.path.join(chkptPath, sorted(chkpts)[-1]) # load checkpoint of regressor of latest epoch
				print('Running LSTM inference on', video, 'with model', chkpt)
				saver.restore(sess, chkpt)
				for trkID in objDict.keys():
					objType, bbDict, original_size, imgDict = objDict[trkID]
					minFrame = min(imgDict.keys())
					maxFrame = max(imgDict.keys())
					length = maxFrame-minFrame+1
					if length<steps:
						continue
					increment = int(windowSize*(1-windowOverlap))-1
					beginFrame = minFrame
					endFrame = beginFrame+windowSize-1
					while endFrame<=maxFrame:
						fRange = range(beginFrame, endFrame+1)
						bbSubDict = {f: bbDict[f] for f in fRange if f in bbDict.keys()}
						ret = sess.run([top_class, top_softmax, loss, fc1, result], feed_dict=form_feedDict(imgDict, original_size, fRange, bbSubDict))
						for i in range(len(ret[0][0])):
							if ret[0][0][i]==1:
								actID += 1
								actDict[actID] = [bbSubDict, objType, trkID, activity, ret[1][0][i]]
						beginFrame += increment
						endFrame += increment

			# Create output files: CSV, JSON, PICKLE
			if not os.path.exists(outFolder):
				os.makedirs(outFolder)
			with open(os.path.join(outFolder, video+'.pickle'), 'wb') as actDictFile:
				pickle.dump(actDict, actDictFile)
			with open(os.path.join(outFolder, video+'.csv'), 'w') as csvFile, open(os.path.join(outFolder, video+'.json'), 'w') as jsonFile:
				jsonData = {u'filesProcessed': [video+'.mp4'], u'activities':[]}
				for actID in actDict.keys():
					bbDict, objType, trkID, activity, conf = actDict[actID]
					beginFrame,endFrame = min(bbDict.keys()),max(bbDict.keys())
					tube = {}
					for f in bbDict.keys():
						x1,y1,w,h = bbDict[f]
						tube[str(f)] = {'boundingBox': {'h': h, 'w': w, 'x': x1, 'y': y1}}
					actIDs = list(str(actID))
					actIDs[0] = 'V'
					csvFile.write(','.join(str(v) for v in [''.join(actIDs), objType, trkID, beginFrame, endFrame, activity, conf])+'\n')
					jsonDict = {
						'activity': activity,
						'activityID': actID,
						'presenceConf': float(conf),
						'alertFrame': endFrame,
						'localization': {
							video+'.mp4': {
								beginFrame: 1,
								endFrame: 0
							}
						},
						'objects': [{
							'localization': {video+'.mp4': tube},
							'objectID': trkID, 
							'objectType': objType
						}]
					}
					jsonData[u'activities'].append(jsonDict)
				json.dump(jsonData, jsonFile, indent=2)
			print('Created', outFolder+'/'+video, 'of', len(actDict), 'activities')