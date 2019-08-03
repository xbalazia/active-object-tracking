import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op, dtypes
import numpy as np
import sys, os, argparse, configparser, cv2, pickle, json

def loadINI(iniPath, i):
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(iniPath)
	return config[i]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', default='config.ini', help='Path to config file.')
	parser.add_argument('-i', '--index', default='0', help='[i] in config file.')
	parser.add_argument('-g', '--gpu', default=0, help='GPU number to use.')
	parser.add_argument('-v', '--videoListPath', dest='videoListPath', default='vids.txt', help='Path to video list file.')
	return parser.parse_args()

def readImages(videoFilePath, trkDictFilePath, size):
	with open(trkDictFilePath, 'rb') as trkDictFile:
		trkDict = pickle.load(trkDictFile)
	objDict = {}
	cap = cv2.VideoCapture(videoFilePath)
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
			objType, bbDict = trkDict[trkID]
			if f not in bbDict.keys():
				continue
			x1,y1,w,h = bbDict[f]
			if trkID not in objDict.keys():
				objDict[trkID] = objType, bbDict, {}
			_,_,imgDict = objDict[trkID]
			imgDict[f] = cv2.resize(cv2.getRectSubPix(image, (w,h), (int(x1+0.5*w),int(y1+0.5*h))), size)
		ok, image = cap.read()
		f += 1
	sys.stdout.write('\n')
	return objDict

def formFrameStack(imgDict, fRange):
	images = []
	imgList = sorted([f for f in imgDict.keys() if f in fRange])
	for f in np.array(imgList)[np.linspace(0, len(imgList), steps, endpoint=False, dtype=np.int32).tolist()].tolist():
		images.append(imgDict[f])
	return np.stack(images)

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
	outDesc = detector+'+'+tracker+'+'+regressor+'_m5'###
	inFolder = os.path.join(tracksFolder, inDesc)
	outFolder = os.path.join(activitiesFolder, outDesc)
	chkptPath = os.path.join('lstm', 'chkpt', varDict['chkptName5'])###
	activities = varDict['activities'].split(',')
	avgWindowSizes = list(map(float, varDict['avgWindowSizes'].split(',')))
	maxWindowSize = 175
	windowOverlap = float(varDict['windowOverlap'])
	classes = 2
	
	inputs = tf.placeholder(tf.float32, [None, input_width, input_height, num_channels])
	ys = tf.placeholder(tf.int64, [None, 1])
	learning_rate = tf.placeholder(tf.float32, [])
	is_training = tf.placeholder(tf.bool)

	ys_one_hot = tf.one_hot(ys, classes)
	
	# Define LSTM
	init_state1 = vs.get_variable('s1', shape=[2*n_hidden1])
	W_lstm1 = vs.get_variable('W1', shape=[feature_size+n_hidden1+n_hidden1, 4*n_hidden1])
	b_lstm1 = vs.get_variable('b1', shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))

	curr_state1 = broadcast(init_state1, [1, 2*n_hidden1])
	_, out_lstm1 = array_ops.split(value=curr_state1, num_or_size_splits=2, axis=1)

	init_state2 = vs.get_variable('s2', shape=[2*n_hidden1])
	W_lstm2 = vs.get_variable('W2', shape=[n_hidden1+n_hidden1, 4*n_hidden1])
	b_lstm2 = vs.get_variable('b2', shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))

	curr_state2 = broadcast(init_state2, [1, 2*n_hidden1])
	_, out_lstm2 = array_ops.split(value=curr_state2, num_or_size_splits=2, axis=1)

	fc_conv_padding = 'VALID'

	inputs = tf.placeholder(tf.float32, (None, 224, 224, 3), name='inputs')
	b, g, r = tf.split(axis=3, num_or_size_splits=3, value=inputs*255.0)
	VGG_MEAN = [103.939, 116.779, 123.68]
	inputs = tf.concat(values=[b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], axis=3)

	with tf.variable_scope('vgg_16', [inputs]) as sc:
		end_points_collection = sc.original_name_scope+'_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		slim = tf.contrib.slim
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=False, scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=False, scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=False, scope='conv5')
			net = slim.max_pool2d(net, [2, 2], scope='pool5')

			# Use conv2d instead of fully_connected layers.
			net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			vgg16_Features = tf.reshape(net, (-1, 4096))
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

	for i in range(steps):
		lstm_input = array_ops.concat([tf.reshape(vgg16_Features[i,:], (-1, feature_size)), out_lstm2], 1)
		#lstm_input = tf.cond(is_training, lambda: tf.nn.dropout(lstm_input, keep_prob = dropout_keep_prob), lambda: lstm_input)

		h_1, curr_state1 = lstm_cell(W_lstm1, b_lstm1, 1.0, lstm_input, curr_state1)
		h_1 = tf.cond(is_training, lambda: tf.nn.dropout(h_1, keep_prob=dropout_keep_prob), lambda: h_1)

		out_lstm1 = h_1

		h_2, curr_state2 = lstm_cell(W_lstm2, b_lstm2, 1.0, out_lstm1, curr_state2)
		h_2 = tf.cond(is_training, lambda: tf.nn.dropout(h_2, keep_prob=dropout_keep_prob), lambda: h_2)
		out_lstm2 = h_2

	W_fc1 = tf.Variable(tf.truncated_normal([n_hidden1, classes], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[classes]))

	fc1 = tf.matmul(out_lstm2,W_fc1)+b_fc1

	# Loss function
	softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=fc1)
	loss = tf.reduce_mean(softmax_loss)

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
			print('\nLSTM regression on '+video)
			actID = 1000000*int('1'+video.replace('VIRAT_S_','').replace('_',''))
			csvList = []
			actDict = {}
			objDict = readImages(os.path.join(videosFolder, video+'.mp4'), os.path.join(inFolder, video+'.pickle'), (input_width,input_height))
			for a in range(len(activities)):
				activity = activities[a]
				print(videoListPath.replace('.txt',''), video, activity)
				chkpts = [chkptFile.replace('.index','') for chkptFile in os.listdir(chkptPath) if chkptFile.startswith(activity+'-') and chkptFile.endswith('.index')]
				chkpt = os.path.join(chkptPath, sorted(chkpts)[-1]) # load checkpoint of regressor trained on activity on latest epoch
				saver.restore(sess, chkpt)
				for trkID in objDict.keys():
					objType, bbDict, imgDict = objDict[trkID]
					actPerson = objType in ['person'] and activity in ['Entering','Exiting','Opening','Closing','Loading','Unloading','Open_Trunk','Closing_Trunk','activity_carrying','Transport_HeavyCarry','Pull','Riding','Talking','specialized_talking_phone','specialized_texting_phone']
					actVehicle = objType in ['car','bus'] and activity in ['Entering','Exiting','Opening','Closing','Loading','Unloading','Open_Trunk','Closing_Trunk','vehicle_turning_left','vehicle_turning_right','vehicle_u_turn']
					if not actPerson and not actVehicle:
						continue
					minFrame = min(imgDict.keys())
					maxFrame = max(imgDict.keys())
					length = maxFrame-minFrame+1
					if length<steps:
						continue
					nSegments = round(length/avgWindowSizes[a])
					if nSegments==0:
						nSegments = 1
					windowSize = round(length/nSegments)
					if windowSize>maxWindowSize:
						windowSize = maxWindowSize
					increment = int(windowSize*(1-windowOverlap))-1
					beginFrame = minFrame
					endFrame = beginFrame+windowSize-1
					while endFrame<=maxFrame:
						actID += 1
						fRange = range(beginFrame, endFrame+1)
						x_train = formFrameStack(imgDict, fRange)
						ret = sess.run([top_class, top_softmax, loss, fc1, result], feed_dict={inputs: np.stack(x_train), ys: np.reshape([0], (-1,1)), is_training: False})
						conf = ret[1][0][ret[0][0][1]]
						bbSubDict = {f: bbDict[f] for f in fRange if f in bbDict.keys()}
						actDict[actID] = [bbSubDict, objType, trkID, activity, conf]
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
						}
					}
					jsonData[u'activities'].append(jsonDict)
				json.dump(jsonData, jsonFile, indent=2)
