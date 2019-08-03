import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op, dtypes
import numpy as np
import sys, os, argparse, configparser, math, random, cv2, pickle, json, resource
# print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

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
	parser.add_argument('-a', '--activity', dest='activity', default='Riding', help='Activity to train.')
	return parser.parse_args()

def gtScan(gtFolder):
	gtDict = {}
	gtJson = os.path.join(gtFolder, 'activities.json')
	if not os.path.exists(gtJson):
		print(gtJson+' does not exist.')
		return {}
	with open(gtJson, 'r') as gtJsonFile:
		gt = json.load(gtJsonFile)
	for videoName in gt['filesProcessed']:
		gtDict[videoName] = {}
	for act in gt['activities']:
		actID = act['activityID']
		actType = act['activity']
		videoName = list(act['localization'].keys())[0]
		beg,end = sorted(list(map(int, act['localization'][videoName].keys())))
		bb = {}
		for f in range(beg,end):
			bb[f] = {'x1': math.inf, 'y1': math.inf, 'x2': -math.inf, 'y2': -math.inf}
			for obj in act['objects']:
				fMax = f
				while str(fMax) not in obj['localization'][videoName].keys() and fMax>=beg:
					fMax -= 1
				if str(fMax) in obj['localization'][videoName].keys() and bool(obj['localization'][videoName][str(fMax)]):
					boundingBox = obj['localization'][videoName][str(fMax)]['boundingBox']
					x1 = min(bb[f]['x1'], int(boundingBox['x']))
					y1 = min(bb[f]['y1'], int(boundingBox['y']))
					x2 = max(bb[f]['x2'], int(boundingBox['x'])+int(boundingBox['w']))
					y2 = max(bb[f]['y2'], int(boundingBox['y'])+int(boundingBox['h']))
					bb[f] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
		bbDict = {}
		for f in range(beg,end):
			bbDict[f] = [bb[f]['x1'],bb[f]['y1'],bb[f]['x2']-bb[f]['x1'],bb[f]['y2']-bb[f]['y1']]
		gtDict[videoName][actID] = actType, bbDict
	return gtDict

def extractImages(gtDict, videosFolder, imgFolder, size):
	for videoName in gtDict.keys():
		print('Extracting GT images from '+videoName+' to '+imgFolder)
		cap = cv2.VideoCapture(os.path.join(videosFolder, videoName))
		try:
			ok, image = cap.read()
		except:
			print('Error reading video '+videoName+'.')
			return {}
		f = 1
		while ok:
			sys.stdout.write('\rFrame: '+str(f))
			sys.stdout.flush()
			for actID in gtDict[videoName].keys():
				actFolder = os.path.join(imgFolder, str(actID))
				if not os.path.isdir(actFolder):
					os.makedirs(actFolder)
				actType, bbDict = gtDict[videoName][actID]
				if f not in bbDict.keys():
					continue
				x1,y1,w,h = bbDict[f]
				overlay = image.copy()
				overlay = cv2.getRectSubPix(image, (w,h), (int(x1+0.5*w),int(y1+0.5*h)))
				overlay = cv2.resize(overlay, size)
				cv2.imwrite(os.path.join(actFolder, str(f)+'.png'), overlay)
			ok, image = cap.read()
			f += 1
		sys.stdout.write('\n')

def getIDs(gtDict, activity):
	actsPerson = ['Entering','Exiting','Opening','Closing','Loading','Unloading','Open_Trunk','Closing_Trunk','activity_carrying','Transport_HeavyCarry','Pull','Riding','Talking','specialized_talking_phone','specialized_texting_phone']
	actsVehicle = ['Entering','Exiting','Opening','Closing','Loading','Unloading','Open_Trunk','Closing_Trunk','vehicle_turning_left','vehicle_turning_right','vehicle_u_turn']
	yDict = {0: [], 1: []}
	for videoName in gtDict:
		for actID in gtDict[videoName]:
			actType, bbDict = gtDict[videoName][actID]
			if set([actType, activity]).issubset(actsPerson) or set([actType, activity]).issubset(actsVehicle):
				yDict[int(actType==activity)].append(actID)
	while len(yDict[0])>len(yDict[1]):
		yDict[0].pop(random.randint(0, len(yDict[0])-1))
	while len(yDict[1])>len(yDict[0]):
		yDict[1].pop(random.randint(0, len(yDict[1])-1))
	return yDict

def formFrameStack(imgFolder):
	images = []
	imgFileNameList = [str(f)+'.png' for f in sorted([int(imgFileName.replace('.png','')) for imgFileName in os.listdir(imgFolder)])]
	for imgFileName in np.array(imgFileNameList)[np.linspace(0, len(imgFileNameList), steps, endpoint=False, dtype=np.int32).tolist()].tolist():
		images.append(cv2.imread(os.path.join(imgFolder, imgFileName)))
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
	activity = args.activity
	varDict = loadINI(args.config, args.index)
	input_width,input_height = int(varDict['input_width']),int(varDict['input_height'])
	num_channels = int(varDict['num_channels'])
	n_hidden1,n_hidden2 = int(varDict['n_hidden1']),int(varDict['n_hidden2'])
	feature_size = int(varDict['feature_size'])
	steps = int(varDict['steps'])
	n_epochs = int(varDict['n_epochs'])
	valid_freq = int(varDict['valid_freq'])
	dropout_keep_prob = float(varDict['dropout_keep_prob'])
	videosFolder = varDict['videosFolder']
	gtFolderTrain = varDict['gtFolderTrain']
	gtFolderValid = varDict['gtFolderValid']
	imgFolderTrain = os.path.join(gtFolderTrain, 'images')
	imgFolderValid = os.path.join(gtFolderValid, 'images')
	chkptPath = os.path.join('lstm', 'chkpt', varDict['chkptName3'])
	if not os.path.exists(chkptPath):
		os.makedirs(chkptPath)
	chkpt = os.path.join(chkptPath, activity)
	classes = 2
	
	# Form train and validation dictionaries
	trainDict = gtScan(gtFolderTrain)
	validDict = gtScan(gtFolderValid)
	if not os.path.isdir(imgFolderTrain):
		extractImages(trainDict, videosFolder, imgFolderTrain, (input_width,input_height))
	if not os.path.isdir(imgFolderValid):
		extractImages(validDict, videosFolder, imgFolderValid, (input_width,input_height))
	trainIDs = getIDs(trainDict, activity)
	validIDs = getIDs(validDict, activity)
	
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
			net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			vgg16_Features = tf.reshape(net, (-1, 4096))
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

	for i in range(steps):
		lstm_input = array_ops.concat([tf.reshape(vgg16_Features[i,:], (-1, feature_size)), out_lstm2], 1)

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

	# Optimization
	#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# Evaluation
	result = tf.argmax(fc1, 1)
	ground_truth = tf.reshape(ys, [-1])
	correct_prediction = tf.equal(result, ground_truth)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg_16'))

	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver(max_to_keep=0)

		# Training
		print('')
		print('Train: positive=%d negative=%d'%(len(trainIDs[1]), len(trainIDs[0])))
		print('Valid: positive=%d negative=%d'%(len(validIDs[1]), len(validIDs[0])))
		epoch = 0
		while epoch<n_epochs:
			if epoch==0:
				learningRate = 1e-3
			if epoch==5:
				learningRate = 1e-4
			if epoch==10:
				learningRate = 1e-5
			epoch += 1
			lossTrain = []
			accTrain = []
			batchTrain = [(y_train, actID) for y_train, actIDs in trainIDs.items() for actID in actIDs]
			random.shuffle(batchTrain)
			for y_train, actID in batchTrain:
				x_train = formFrameStack(os.path.join(imgFolderTrain, str(actID)))
				ret = sess.run([train_op, loss, accuracy], feed_dict={inputs: x_train, ys: np.reshape([y_train], (-1,1)), is_training: True, learning_rate: learningRate})
				lossTrain.append(ret[1])
				accTrain.append(ret[2])
			string = 'TRAIN activity=%s epoch=%03d learningRate=%.0e loss=%.3f accuracy=%.3f'%(activity, epoch, learningRate, np.mean(lossTrain), np.mean(accTrain))
			print(string)
			with open(chkpt+'.log', 'a') as logFile:
				logFile.write(string+'\n')

			# Run validation and save model
			if epoch%valid_freq==0:
				lossValid = []
				accValid = []
				batchValid = [(y_valid, actID) for y_valid, actIDs in validIDs.items() for actID in actIDs]
				for y_valid, actID in batchValid:
					x_valid = formFrameStack(os.path.join(imgFolderValid, str(actID)))
					ret = sess.run([result, loss, accuracy], feed_dict={inputs: x_valid, ys: np.reshape([y_valid], (-1,1)), is_training: False})
					lossValid.append(ret[1])
					accValid.append(ret[2])
				string = 'VALID activity=%s epoch=%03d loss=%.3f accuracy=%.3f'%(activity, epoch, np.mean(lossValid), np.mean(accValid))
				print(string)
				with open(chkpt+'.log', 'a') as logFile:
					logFile.write(string+'\n')
				saver.save(sess, chkpt+'-'+str(epoch))
