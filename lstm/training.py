import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op, dtypes
import numpy as np
import sys, os, argparse, configparser, math, random, cv2, json, pickle, shutil, resource
from datetime import datetime, timedelta
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
	parser.add_argument('-m', '--model', default='m0', help='Model identifier.')
	parser.add_argument('-a', '--activity', default='activity_carrying', help='Activity name.')
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
	#return [[0,0]]

def getNormalizedBoundingBoxInfo(bbDict, original_size):
	original_width, original_height = original_size
	bbInfo = []
	for f in bbDict.keys():
		x1,y1,w,h = bbDict[f]
		x0,y0 = x1+0.5*w, y1+0.5*h
		bbInfo.append([x0/original_width, y0/original_height, w/original_width, h/original_height])
		#bbInfo.append([0,0,0,0])
	return linSample(bbInfo)

def isOverlapping(bbA, bbB):
	x1A,y1A,wA,hA = bbA
	x1B,y1B,wB,hB = bbB
	# if one rectangle is on left side of other 
	if x1B+wB<x1A or x1A+wA<x1B:
		return False
	# if one rectangle is above other 
	if y1B+hB<y1A or y1A+hA<y1B:
		return False
	return True

def form_gtDict(gtFolder, activity):
	gtDictFilePath = os.path.join(gtFolder, 'dict.pickle')
	if not os.path.isfile(gtDictFilePath):
		# If gtDict does not exist, load ground truth
		print('Forming GT dictionary from', gtFolder)
		gtJson = os.path.join(gtFolder, 'activities.json')
		if not os.path.exists(gtJson):
			print(gtJson+' does not exist.')
			return {}
		with open(gtJson, 'r') as gtJsonFile:
			gt = json.load(gtJsonFile)
		# Form gtDict = {actNum: {actID: [videoName, bbDict, travel]}}
		gtDict = {}
		for act in gt['activities']:
			activity = act['activity']
			actID = act['activityID']
			videoName = list(act['localization'].keys())[0]
			beg, end = sorted(list(map(int, act['localization'][videoName].keys())))
			bbDict = {}
			for f in range(beg, end):
				bbDict[f] = [math.inf, math.inf, -math.inf, -math.inf]
				for obj in act['objects']:
					fMax = f
					while str(fMax) not in obj['localization'][videoName].keys() and fMax>=beg:
						fMax -= 1
					if str(fMax) in obj['localization'][videoName].keys() and bool(obj['localization'][videoName][str(fMax)]):
						boundingBox = obj['localization'][videoName][str(fMax)]['boundingBox']
						x1,y1,w,h = bbDict[f]
						x1 = min(x1, int(boundingBox['x']))
						y1 = min(y1, int(boundingBox['y']))
						x2 = max(x1+w, int(boundingBox['x'])+int(boundingBox['w']))
						y2 = max(y1+h, int(boundingBox['y'])+int(boundingBox['h']))
						bbDict[f] = [x1,y1,x2-x1,y2-y1]
			actNum = activities.index(activity)
			if actNum not in gtDict.keys():
				gtDict[actNum] = {}
			gtDict[actNum][actID] = [videoName, bbDict, getNormalizedTravelInfo(bbDict)]
		# Add background class
		bgNumPerson = activities.index(bgLabelPerson)
		bgNumCar = activities.index(bgLabelCar)
		gtDict[bgNumPerson] = {}
		gtDict[bgNumCar] = {}
		bgID = actID
		objTypeDict = {}
		minNormTravelIndex = 0.65
		minNormAbsTravelAngle = 0.25
		minLength = 50
		maxLength = 100
		for videoName in gt['filesProcessed']:
			with open(os.path.join(tracksFolder, videoName.replace('.mp4', '.pickle')), 'rb') as trkDictFile:
				trkDict = pickle.load(trkDictFile)
			for trkID in trkDict.keys():
				objType, bbDict = trkDict[trkID]
				fList = []
				for f in range(min(bbDict.keys()), max(bbDict.keys())+1):
					overlap = False
					for actNum in gtDict.keys():
						for actID in gtDict[actNum].keys():
							if actNum not in [bgNumPerson, bgNumCar] and f in gtDict[actNum][actID][1].keys():
								if isOverlapping(bbDict[f], gtDict[actNum][actID][1][f]):
									overlap = True
									break
						if overlap:
							break
					if not overlap:
						fList.append(f)
				bDict = {}
				for f in fList:
					if bool(bDict) and f-max(bDict.keys())>=maxLength:
						bDict[f] = [f]
					else:
						k = f
						while k-1 in fList and k not in bDict.keys():
							k -= 1
						if k not in bDict.keys():
							bDict[k] = []
						bDict[k].append(f)
				for k in bDict.keys():
					bbSubDict = {f:v for f,v in bbDict.items() if f in bDict[k]}
					travelInfo = getNormalizedTravelInfo(bbSubDict)
					if minLength<=len(bbSubDict) and minNormTravelIndex<=travelInfo[0][0] and minNormAbsTravelAngle<=abs(0.5-travelInfo[0][1]):
						bgID += 1
						sys.stdout.write('\r'+str(bgID)+' '+videoName)
						sys.stdout.flush()
						if objType=='person':
							bgNum = bgNumPerson
						if objType=='car':
							bgNum = bgNumCar
						gtDict[bgNum][bgID] = [videoName, bbSubDict, travelInfo]
		# Save gtDict into pickle
		with open(gtDictFilePath, 'wb') as gtDictFile:
			pickle.dump(gtDict, gtDictFile)
		# Extract ground truth images
		print('\nExtracting ground truth images')
		imgFolder = os.path.join(gtFolder, 'images')
		if os.path.isdir(imgFolder):
			shutil.rmtree(imgFolder)
		os.makedirs(imgFolder)
		for videoName in gt['filesProcessed']:
			cap = cv2.VideoCapture(os.path.join(videosFolder, videoName))
			try:
				ok, image = cap.read()
			except:
				print('Error reading video '+videoName+'.')
				return {}
			f = 1
			while ok:
				sys.stdout.write('\r'+videoName+': Frame: '+str(f))
				sys.stdout.flush()
				for actNum in gtDict.keys():
					for actID in gtDict[actNum].keys():
						videoName_this, bbDict_this, _ = gtDict[actNum][actID]
						if videoName!=videoName_this or f not in bbDict_this.keys():
							continue
						actFolder = os.path.join(imgFolder, str(actID))
						if not os.path.isdir(actFolder):
							os.makedirs(actFolder)
						x1,y1,w,h = bbDict_this[f]
						overlay = image.copy()
						overlay = cv2.getRectSubPix(image, (w,h), (int(x1+0.5*w),int(y1+0.5*h)))
						overlay = cv2.resize(overlay, (input_width, input_height))
						cv2.imwrite(os.path.join(actFolder, str(f)+'.png'), overlay)
				ok, image = cap.read()
				f += 1
			sys.stdout.write('\n')
	else:
		# If gtDict exists, load gtDict from pickle
		print('Loading GT dictionary from', gtFolder)
		with open(gtDictFilePath, 'rb') as gtDictFile:
			gtDict = pickle.load(gtDictFile)
	# Print statistics
	for actNum in sorted(gtDict.keys()):
		print('%s=%d'%(activities[actNum], len(gtDict[actNum])))
	# Modify gtDict into binary
	pool1 = ['Entering','Exiting','Loading','Unloading','Open_Trunk','Closing_Trunk','Opening','Closing']
	pool2 = ['activity_carrying','Transport_HeavyCarry','Pull','Riding','Talking','specialized_talking_phone','specialized_texting_phone']
	pool3 = ['vehicle_turning_left','vehicle_turning_right','vehicle_u_turn']
	for pool in [pool1,pool2,pool3]:
		if activity in pool:
			dict1 = gtDict[activities.index(activity)]
			dict0 = {}
			for activity0 in pool:
				if activity0==activity:
					if pool in [pool2]:
						bgLabel = bgLabelPerson
					if pool in [pool1, pool3]:
						bgLabel = bgLabelCar
					dict0.update(gtDict[activities.index(bgLabel)])
				else:
					dict0.update(gtDict[activities.index(activity0)])
	# Sample down to ensure equal sizes of dict0 and dict1
	if len(dict0)>len(dict1):
		for actID in random.sample(dict0.keys(), len(dict0)-len(dict1)):
			del dict0[actID]
	if len(dict1)>len(dict0):
		for actID in random.sample(dict1.keys(), len(dict1)-len(dict0)):
			del dict1[actID]
	# Return binDict
	return {0: dict0, 1: dict1}

def linSample(array):
	return np.array(array)[np.linspace(0, len(array), steps, endpoint=False, dtype=np.int32).tolist()]

def form_feedDict(gtFolder, binNum, actID, actInfo, lr):
	videoName, bbDict, _ = actInfo
	cap = cv2.VideoCapture(os.path.join(videosFolder, videoName))
	original_size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	imgFolder = os.path.join(gtFolder, 'images', str(actID))
	imgList = []
	for imgNum in linSample([str(f)+'.png' for f in sorted([int(imgNum.replace('.png','')) for imgNum in os.listdir(imgFolder)])]):
		imgList.append(cv2.imread(os.path.join(imgFolder, imgNum)))
	fd_label = np.reshape([binNum], (-1, 1))
	fd_images = np.stack(imgList)
	fd_travel = getNormalizedTravelInfo(bbDict)
	fd_bbox = getNormalizedBoundingBoxInfo(bbDict, original_size)
	fd_is_training = lr==0
	return {label: fd_label, images: fd_images, travel: fd_travel, bbox: fd_bbox, is_training: fd_is_training, learning_rate: lr}

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
	return tensor + tf.zeros(shape, dtype=tensor.dtype)

# Main
if __name__ == '__main__':

	# Load arguments and variables from config file
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	varDict = loadINI(args.config, args.index)
	input_width, input_height = int(varDict['input_width']), int(varDict['input_height'])
	num_channels = int(varDict['num_channels'])
	n_hidden1, n_hidden2 = int(varDict['n_hidden1']), int(varDict['n_hidden2'])
	feature_size = int(varDict['feature_size'])
	steps = int(varDict['steps'])
	n_epochs = int(varDict['n_epochs'])
	valid_freq = int(varDict['valid_freq'])
	dropout_keep_prob = float(varDict['dropout_keep_prob'])
	videosFolder = varDict['videosFolder']
	tracksFolder = inFolder = os.path.join(varDict['tracksFolder'], varDict['detector']+'+'+varDict['tracker'])
	trainFolder = varDict['trainFolder']
	validFolder = varDict['validFolder']
	bgLabelPerson = varDict['bgPerson']
	bgLabelCar = varDict['bgCar']
	activities = varDict['activities'].split(',')
	activities.append(bgLabelPerson)
	activities.append(bgLabelCar)
	activity = args.activity
	if activity not in activities:
		print(activity, 'not in activities.')
		sys.exit()
	chkptPath = os.path.join('lstm', 'chkpt', 'vgg16-'+args.model)
	if not os.path.exists(chkptPath):
		os.makedirs(chkptPath)
	chkpt = os.path.join(chkptPath, activity)
	classes = 2

	# Form train and validation dictionaries
	print('')
	trainDict = form_gtDict(trainFolder, activity)
	print('')
	validDict = form_gtDict(validFolder, activity)
	print('')
	
	# Define LSTM
	label = tf.placeholder(tf.int64, [None, 1])
	images = tf.placeholder(tf.float32, [None, input_width, input_height, num_channels])
	travel = tf.placeholder(tf.float32, [None, 2])
	bbox = tf.placeholder(tf.float32, [None, 4])
	is_training = tf.placeholder(tf.bool)
	learning_rate = tf.placeholder(tf.float32, [])

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

	# Optimization
	#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# Evaluation
	result = tf.argmax(fc1, 1)
	ground_truth = tf.reshape(label, [-1])
	correct_prediction = tf.equal(result, ground_truth)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=0)

	with tf.Session() as sess:
		sess.run(init)

		# Training
		print('')
		epoch = 0
		while epoch<n_epochs:
			if epoch==0:
				lr = 1e-4
			epoch += 1
			lossTrain = []
			accTrain = []
			batchTrain = [(binNum, actID, actInfo) for binNum, actDict in trainDict.items() for actID, actInfo in actDict.items()]
			random.shuffle(batchTrain)
			b = 0
			start = datetime.now()
			for binNum, actID, actInfo in batchTrain:
				b += 1
				sys.stdout.write('\rTraining epoch %d: %d/%d'%(epoch, b, len(batchTrain)))
				sys.stdout.flush()
				ret = sess.run([train_op, loss, accuracy], feed_dict=form_feedDict(trainFolder, binNum, actID, actInfo, lr))
				lossTrain.append(ret[1])
				accTrain.append(ret[2])
			time = datetime.now()-start
			string = 'TRAIN activity=%s epoch=%03d learningRate=%.0e loss=%.3f accuracy=%.3f time=%dmin'%(activity, epoch, lr, np.mean(lossTrain), np.mean(accTrain), int(time.seconds/60))
			sys.stdout.write('\r'+string+'\n')
			with open('%s.log'%(chkpt), 'a') as logFile:
				logFile.write(string+'\n')

			# Run validation and save model
			if epoch%valid_freq==0:
				lossValid = []
				accValid = []
				batchValid = [(binNum, actID, actInfo) for binNum, actDict in validDict.items() for actID, actInfo in actDict.items()]
				b = 0
				start = datetime.now()
				for binNum, actID, actInfo in batchValid:
					b += 1
					sys.stdout.write('\rValidation epoch %d: %d/%d'%(epoch, b, len(batchValid)))
					sys.stdout.flush()
					ret = sess.run([result, loss, accuracy], feed_dict=form_feedDict(validFolder, binNum, actID, actInfo, 0))
					lossValid.append(ret[1])
					accValid.append(ret[2])
				time = datetime.now()-start
				string = 'VALID activity=%s epoch=%03d loss=%.3f accuracy=%.3f time=%dmin'%(activity, epoch, np.mean(lossValid), np.mean(accValid), int(time.seconds/60))
				sys.stdout.write('\r'+string+'\n\n')
				with open('%s.log'%(chkpt), 'a') as logFile:
					logFile.write(string+'\n')
				saver.save(sess, '%s-%03d'%(chkpt, epoch))
