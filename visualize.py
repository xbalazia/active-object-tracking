import sys, os, argparse, configparser, math, cv2, json, pickle
import numpy as np

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
	parser.add_argument('-t', '--types', dest='types', default='det,trk,act1,act2,act3', help='Types to visualize.')
	return parser.parse_args()

def visualize(videos, inPath, outPath, visInfo):
	"""
	http://www.fourcc.org/codecs.php
	http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
	"""
	for video in videos:
		inVidPath = os.path.join(inPath, video+'.mp4')
		outVidPath = os.path.join(outPath, video+'.mp4')
		outImgPath = os.path.join(outPath, video+'.png')
		if not os.path.isfile(inVidPath):
			print(inVidPath+' does not exist.')
			continue
		if not os.path.exists(os.path.dirname(outVidPath)):
			os.makedirs(os.path.dirname(outVidPath))
		if not os.path.exists(os.path.dirname(outImgPath)):
			os.makedirs(os.path.dirname(outImgPath))
		inVid = cv2.VideoCapture(inVidPath)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		fps = int(round(inVid.get(cv2.CAP_PROP_FPS)))
		size = (int(round(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))), int(round(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
		outVid = cv2.VideoWriter(outVidPath, fourcc, fps, size)
		inVid = cv2.VideoCapture(inVidPath)
		try:
			ok, image = inVid.read()
		except:
			print('Error reading video '+inVidPath+'.')
			continue
		visDicts = {}
		for typ in visInfo.keys():
			desc, folder, color = visInfo[typ]
			xxDictFilePath = os.path.join(folder, video+'.pickle')
			if not os.path.exists(xxDictFilePath):
				print('Unable to visualize', desc, 'on', video, 'as', xxDictFilePath, 'does not exist.')
				continue
			with open(xxDictFilePath, 'rb') as xxDictFile:
				visDict = pickle.load(xxDictFile)
			visDicts[typ] = [visDict, color]

		# Make image
		overlay = image.copy()
		for typ in visDicts.keys():
			visDict, color = visDicts[typ]
			for visID in visDict.keys():
				bbDict = visDict[visID][0]
				if len(bbDict)==1:
					for f in bbDict.keys():
						x1,y1,w,h = bbDict[f]
						cv2.circle(overlay, (int(x1+w/2),int(y1+h/2)), 1, color, thickness=2)
				else:
					points = sorted(bbDict.keys())
					for p in range(1,len(points)):
						pt_a = points[p-1]
						pt_b = points[p]
						x1_a,y1_a,w_a,h_a = bbDict[pt_a]
						x1_b,y1_b,w_b,h_b = bbDict[pt_b]
						cv2.line(overlay, (int(x1_a+w_a/2),int(y1_a+h_a/2)), (int(x1_b+w_b/2),int(y1_b+h_b/2)), color, thickness=2)
		cv2.imwrite(outImgPath,overlay)

		# Make video
		f = 1
		images = {}
		while ok:
			sys.stdout.write('\r'+video+': Read '+str(f))
			sys.stdout.flush()
			images[f] = image
			ok, image = inVid.read()
			f += 1
		l = len(images)
		for f in sorted(images.keys()):
			sys.stdout.write('\r'+video+': Write '+str(f)+'/'+str(l))
			sys.stdout.flush()
			image = images[f]
			for typ in visDicts.keys():
				visDict, color = visDicts[typ]
				for visID in visDict.keys():
					bbDict, objType, trkID, activity, conf = visDict[visID]
					percentage = '('+str(int(round(conf*100)))+'%)'
					if f in bbDict.keys():
						x1,y1,w,h = bbDict[f]
						if typ == 'gt':
							desc = str(visID)+': '+activity
						if typ == 'det':
							desc = objType+' '+percentage
						if typ == 'trk':
							desc = str(visID)+': '+objType+' '+percentage
						if typ == 'act1':
							desc = objType+' '+percentage
						if typ == 'act2':
							desc = str(visID%1000000)+': '+activity+' '+percentage
						if typ == 'act3':
							desc = activity+' '+percentage
						opacity = conf # math.pow(c, 2) # caption and box opacity = confidence squared
						overlay = image.copy()
						cv2.putText(overlay, desc, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color)
						cv2.rectangle(overlay, (x1,y1), (x1+w,y1+h), color, 1)
						cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)
			outVid.write(image)
		sys.stdout.write('\n')
		outVid.release()

if __name__ == '__main__':

	# Load config and get variables
	args = parse_args()
	videoListPath = args.videoListPath
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	varDict = loadINI(args.config, args.index)
	gtFolders = [varDict['trainFolder'], varDict['validFolder']]
	detFolder = varDict['detectionsFolder']
	trkFolder = varDict['tracksFolder']
	actFolder = varDict['activitiesFolder']
	objects = varDict['objects'].split(',')
	detector = varDict['detector']
	tracker = varDict['tracker']
	regressor = varDict['regressor']+'_'+args.model
	filterer = varDict['filterer']
	dataFolder = varDict['dataFolder']
	videosFolder = varDict['videosFolder']
	gtDictFolder = varDict['gtDictFolder']
	visInfo = {}
	gtDesc = 'gt'
	detDesc = detector
	trkDesc = detector+'+'+tracker
	act1Desc = detector+'+'+tracker+'+'+regressor
	act2Desc = detector+'+'+tracker+'+'+regressor+'+'+filterer
	act3Desc = detector+'+'+tracker+'+'+regressor+'+'+filterer+'+max'
	green = (0,255,0)
	red = (0,0,255)
	blue = (255,0,0)
	purple = (255,0,255)
	yellow = (0,255,255)
	white = (255,255,255)
	types = args.types.split(',')
	typeDict = 	{	'gt':	['gt',	gtDesc,		gtDictFolder,						green],
					'det':	['det',	detDesc,	os.path.join(detFolder, detDesc),	red],
					'trk':	['trk',	trkDesc,	os.path.join(trkFolder, trkDesc),	blue],
					'act1':	['act1',	act1Desc,	os.path.join(actFolder, act1Desc),	purple],
					'act2':	['act2',	act2Desc,	os.path.join(actFolder, act2Desc),	yellow],
					'act3':	['act3',	act3Desc,	os.path.join(actFolder, act3Desc),	white],
				}

	# Run visualizations
	for typ in types:
		typ,desc,folder,color = typeDict[typ]
		visInfo[typ] = [desc,folder,color]
	description = '['+','.join([typ+'('+visInfo[typ][0]+')' for typ in visInfo.keys()])+']'
	print('Visualizing', description)
	visualize(videos, videosFolder, os.path.join(dataFolder, 'vis'+description), visInfo)