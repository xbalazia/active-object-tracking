import sys, os, argparse, configparser, cv2, pickle
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from keras_ssd512 import ssd_512
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import decode_y

def loadINI(iniPath, i):
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(iniPath)
	return config[i]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.ini', help='Path to config file.')
	parser.add_argument('--index', default='0', help='[i] in config file.')
	parser.add_argument('--gpu', default='0', help='GPU number to use.')
	parser.add_argument('--video_list_path', dest='video_list_path', help='Path to video list file.', default='vids.txt')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	varDict = loadINI(args.config, args.index)
	videosFolder = varDict['videosFolder']
	detectionsFolder = varDict['detectionsFolder']
	detConfThresh = float(varDict['detConfThresh'])
	weightsFilePath = 'ssd/VGG_coco_SSD_512x512.h5'
	img_width,img_height = 512,512
	detector = 'ssd'
	fill = '        '
	
	# Build the Keras model
	K.clear_session() # Clear previous models from memory.
	model = ssd_512(image_size=(img_height, img_width, 3),
					n_classes=20,
					l2_regularization=0.0005,
					scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06], #scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
					aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
											 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
											 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
											 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
											 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
											 [1.0, 2.0, 0.5],
											 [1.0, 2.0, 0.5]],
					two_boxes_for_ar1=True,
					steps=[8, 16, 32, 64, 128, 256, 512],
					offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
					limit_boxes=False,
					variances=[0.1, 0.1, 0.2, 0.2],
					coords='centroids',
					normalize_coords=True,
					subtract_mean=[123, 117, 104],
					swap_channels=True)
	model.load_weights(weightsFilePath, by_name=True)
	print('Model {} loaded successfully!'.format(weightsFilePath.split('/')[-1].split('.')[0]))

	# Compile the model so that Keras won't complain the next time you load it.
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
	ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
	model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
	orig_images = [] # Store the images here.
	input_images = [] # Store resized versions of the images here.
	classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
	with open(args.video_list_path,'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	for video in videos:
		cap = cv2.VideoCapture(os.path.join(videosFolder, video+'.mp4'))
		f = 0
		detID = ''
		detDict = {detID: []}
		while(1):
			ret, frame = cap.read()
			if not ret:
				break
			orig = image.img_to_array(frame)
			f += 1
			resized_image = cv2.resize(frame, (img_width, img_height))
			img = image.img_to_array(resized_image)	
			input_images.append(img)
			input_images = np.array(input_images)
			y_pred = model.predict(input_images)
			y_pred_decoded = decode_y(y_pred,
							  confidence_thresh=detConfThresh,
							  iou_threshold=0.45,
							  top_k=200,
							  input_coords='centroids',
							  normalize_coords=True,
							  img_height=img_height,
							  img_width=img_width)
			for class_id, confidence, xmin, ymin, xmax, ymax in y_pred_decoded[0]:
				objType = classes[int(class_id)]
				# Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
				x1 = int(xmin * orig.shape[1] / img_width)
				y1 = int(ymin * orig.shape[0] / img_height)
				x2 = int(xmax * orig.shape[1] / img_width)
				y2 = int(ymax * orig.shape[0] / img_height)
				conf = round(confidence,4)
				detDict[detID].append([f,objType,x1,y1,x2-x1,y2-y1,conf])
			input_images = []
			sys.stdout.write('\r'+video+': '+str(f)+fill)
			sys.stdout.flush()

		# Create output files
		print('Creating output PICKLE')
		detFolder = os.path.join(detectionsFolder, detector)
		if not os.path.exists(detFolder):
			os.makedirs(detFolder)
		with open(os.path.join(detFolder, video+'.pickle'), 'wb') as detDictFile:
				pickle.dump(detDict, detDictFile)