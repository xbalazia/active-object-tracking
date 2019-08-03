import sys, os, argparse, configparser, math, cv2, json, pickle
import numpy as np

fill = '            '

def loadINI(iniPath, i):
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(iniPath)
	return config[i]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', default='config.ini', help='Path to config file.')
	parser.add_argument('-i', '--index', default='0', help='[i] in config file.')
	parser.add_argument('-v', '--videoListPath', dest='videoListPath', default='vids.txt', help='Path to video list file.')
	parser.add_argument('-d', '--detector', default='ssd', help='Detector.')
	parser.add_argument('-t', '--tracker', default='mosse', help='Tracker.')
	parser.add_argument('-r', '--regressor', default='lstm', help='Regressor.')
	parser.add_argument('-f', '--filterer', default='hcf', help='Filterer.')
	return parser.parse_args()

def loadVideos(vidList):
	if not os.path.isfile(vidList):
		print(vidList+' does not exist.')
		return []
	with open(vidList, 'r') as vidListFile:
		return vidListFile.read().splitlines()

def splitFrames(data):
	splitFrames = {}
	for [actID,actType,f,x1,y1,x2,y2] in data:
		if f not in splitFrames.keys():
			splitFrames[f] = []
		splitFrames[f].append([actID,actType,x1,y1,x2,y2])
	return splitFrames

def iou(boxA,boxB):
	x1A,y1A,x2A,y2A = boxA
	x1B,y1B,x2B,y2B = boxB
	if x1A<x2B and x2A>x1B and y1A<y2B and y2A>y1B:
		xA = max(x1A,x1B)
		yA = max(y1A,y1B)
		xB = min(x2A,x2B)
		yB = min(y2A,y2B)
		iArea = (xB-xA)*(yB-yA)
		boxAArea = (x2A-x1A)*(y2A-y1A)
		boxBArea = (x2B-x1B)*(y2B-y1B)
		uArea = boxAArea+boxBArea-iArea
		return iArea/float(uArea)
	return 0

def gtScan(videos, gtFolders):
	visDict = {}
	for video in videos:
		visDict[video] = []
		videoName = video+'.mp4'
		found = False
		for gtFolder in gtFolders:
			gtJson = os.path.join(gtFolder, video+'.json')
			if os.path.exists(gtJson):
				found = True
				break
		if not found:
			print('GT for '+video+' does not exist.')
			continue
		with open(gtJson, 'r') as gtJsonFile:
			gt = json.load(gtJsonFile)
		for act in gt['activities']:
			actID = act['activityID']
			actType = act['activity']
			beg,end = sorted(list(map(int, act['localization'][videoName].keys())))
			bb = {}
			for f in range(beg,end):
				bb[f] = {'x1':math.inf, 'y1':math.inf, 'x2':-math.inf, 'y2':-math.inf}
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
						bb[f] = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
			for f in range(beg,end):
				visDict[video].append([actID,actType,f,bb[f]['x1'],bb[f]['y1'],bb[f]['x2'],bb[f]['y2'],1])
	return visDict

def detScan(videos, detFolder, objects):
	visDict = {}
	for video in videos:
		visDict[video] = []
		detDictFilePath = os.path.join(detFolder, video+'.pickle')
		if not os.path.exists(detDictFilePath):
			print(detDictFilePath+' does not exist.')
			continue
		with open(detDictFilePath, 'rb') as detDictFile:
			detDict = pickle.load(detDictFile)
		for detID in detDict.keys():
			for f,objType,x1,y1,w,h,conf in detDict[detID]:
				if objType in objects:
					visDict[video].append([detID,objType,f,x1,y1,x1+w,y1+h,conf])
	return visDict

def trkScan(videos, trkFolder, objects):
	visDict = {}
	for video in videos:
		visDict[video] = []
		trkDictFilePath = os.path.join(trkFolder, video+'.pickle')
		if not os.path.exists(trkDictFilePath):
			print(trkDictFilePath+' does not exist.')
			continue
		with open(trkDictFilePath, 'rb') as trkDictFile:
			trkDict = pickle.load(trkDictFile)
		for trkID in trkDict.keys():
			objType, bbDict = trkDict[trkID]
			if objType in objects:
				for f in bbDict.keys():
					x1,y1,w,h = bbDict[f]
					visDict[video].append([trkID,objType,f,x1,y1,x1+w,y1+h,1])
	return visDict

def actScan(videos, actFolder, activities):
	visDict = {}
	for video in videos:
		visDict[video] = []
		actDictFilePath = os.path.join(actFolder, video+'.pickle')
		if not os.path.exists(actDictFilePath):
			print(actDictFilePath+' does not exist.')
			continue
		with open(actDictFilePath, 'rb') as actDictFile:
			actDict = pickle.load(actDictFile)
		for actID in actDict.keys():
			bbDict, objType, trkID, activity, conf = actDict[actID]
			if activity in activities:
				for f in bbDict.keys():
					x1,y1,w,h = bbDict[f]
					visDict[video].append([actID%1000000,activity,f,x1,y1,x1+w,y1+h,conf])
	return visDict

def visXY(inPath, outPath, visDicts):
	"""
	http://www.fourcc.org/codecs.php
	http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
	"""
	for video in visDicts[0][0].keys():
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
		# Make image
		overlay = image.copy()
		for visDict,color in visDicts:
			data = visDict[video]
			if len(data)>0 and data[0][0]=='':
				for i,t,f,x1,y1,x2,y2,c in data:
					x = int((x1+x2)/2)
					y = int((y1+y2)/2)
					cv2.circle(overlay, (x,y), 1, color, thickness=2)
			else:
				dataDict = {}
				for i,t,f,x1,y1,x2,y2,c in data:
					if i not in dataDict.keys():
						dataDict[i] = {}
					dataDict[i][f] = [x1,y1,x2,y2]
				for i in dataDict.keys():
					line = dataDict[i]
					points = sorted(line.keys())
					for p in range(1,len(points)):
						ptA = points[p-1]
						ptB = points[p]
						x1b,y1b,x2b,y2b = line[ptA]
						x1a,y1a,x2a,y2a = line[ptB]
						xa = int((x1a+x2a)/2)
						ya = int((y1a+y2a)/2)
						xb = int((x1b+x2b)/2)
						yb = int((y1b+y2b)/2)
						cv2.line(overlay, (xa,ya), (xb,yb), color, thickness=2)
		cv2.imwrite(outImgPath,overlay)
		# Make video
		f = 1
		images = {}
		while ok:
			sys.stdout.write('\r'+video+': Read '+str(f)+fill)
			sys.stdout.flush()
			images[f] = image
			ok, image = inVid.read()
			f += 1
		l = len(images)
		for f in sorted(images.keys()):
			sys.stdout.write('\r'+video+': Write '+str(f)+'/'+str(l)+fill)
			sys.stdout.flush()
			image = images[f]
			for visDict,color in visDicts:
				data = visDict[video]
				for i,t,fd,x1,y1,x2,y2,c in data:
					if f==fd:
						o = math.pow(c, 2) # caption and box opacity = confidence squared
						overlay = image.copy()
						if c==1:
							percent = ''
						else:
							percent = ' ('+str(int(round(c*100)))+'%)'
						cv2.putText(overlay, str(i)+': '+t+percent, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color)
						cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 1)
						cv2.addWeighted(overlay, o, image, 1-o, 0, image)
			outVid.write(image)
		sys.stdout.write('\n')
		outVid.release()

def visXT(inPath, outPath, visDicts):
	"""
	http://www.fourcc.org/codecs.php
	http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
	"""
	for video in sorted(visDicts[0][0].keys()):
		print(video)
		inVidPath = os.path.join(inPath, video+'.mp4')
		outImgPath = os.path.join(outPath, video)
		if not os.path.isfile(inVidPath):
			print(inVidPath+' does not exist.')
			continue
		if not os.path.exists(outImgPath):
			os.makedirs(outImgPath)
		inVid = cv2.VideoCapture(inVidPath)
		f = 0
		inImages = {}
		ok,inImage = inVid.read()
		while ok:
			sys.stdout.write('\r Read: '+str(f))
			sys.stdout.flush()
			inImages[f] = inImage
			ok,inImage = inVid.read()
			f += 1
		l = len(images)
		outDict = {}
		for visDict,color in visDicts:
			dataDict = {}
			typeDict = {}
			for i,t,f,x1,y1,x2,y2,c in visDict[video]:
				if i not in dataDict.keys():
					dataDict[i] = {}
					typeDict[i] = t
				for y in range(y1,y2+1):
					if y not in dataDict[i].keys():
						dataDict[i][y] = [math.inf,math.inf,0,0]
					x1m,f1m,x2m,f2m = dataDict[i][y]
					dataDict[i][y] = [min(x1,x1m),min(f,f1m),max(x2,x2m),max(f,f2m)]
			for i in dataDict.keys():
				for y in dataDict[i].keys():
					if y not in outDict.keys():
						outDict[y] = []
					x1m,f1m,x2m,f2m = dataDict[i][y]
					outDict[y].append([i,typeDict[i],x1m,f1m,x2m,f2m,color])
		height = int(round(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		for y in range(1,height+1):
			sys.stdout.write('\r Write: '+str(y)+'/'+str(height))
			sys.stdout.flush()
			image = np.zeros((len(inImages), int(round(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))), 3), dtype=np.uint8)
			for f in inImages.keys():
				image[c-1,:] = inImages[f][y-1,:]
			if y in outDict.keys():
				for i,t,x1m,f1m,x2m,f2m,color in outDict[y]:
					cv2.putText(image, str(i)+': '+t, (x1m, f1m), cv2.FONT_HERSHEY_DUPLEX, 0.7, color)
					cv2.rectangle(image, (x1m, f1m), (x2m, f2m), color, 1)
			cv2.imwrite(os.path.join(outImgPath, str(y).zfill(5)+'.png'), image)
			sys.stdout.write('\r')

def coverage(visDictA, visDictB):
	hits = 0
	total = 0
	for video in sorted(visDictA.keys()):
		aSplitFrames = splitFrames(visDictA[video])
		bSplitFrames = splitFrames(visDictB[video])
		for f in aSplitFrames.keys():
			for aID,aType,aX1,aY1,aX2,aY2,aC in aSplitFrames[f]:
				total += 1
				hit = False
				if f in bSplitFrames.keys():
					for bID,bType,bX1,bY1,bX2,bY2,bC in bSplitFrames[f]:
						if iou([aX1,aY1,aX2,aY2],[bX1,bY1,bX2,bY2])>0.1: # >10% overlap
							hit = True
				if hit:
					hits += 1
	return float(100)*hits/total

def fMeasure(gt,trk):
	recall = coverage(gt[0], trk[0])
	precision = coverage(trk[0], gt[0])
	print('recall='+str(recall)+'\n')
	print('precision='+str(precision)+'\n')
	print(precision)
	fMeasure = round((2*recall*precision)/(recall+precision),2)
	print('fMeasure='+str(fMeasure)+'\n')

def duplicate(gt, outFilePath):
	gtDict = gt[0]
	hitsBox = 0
	hitsAct = 0
	totalBox = 0
	totalAct = 0
	duplBox = {}
	duplAct = {}
	ids = []
	for video in sorted(gtDict.keys()):
		duplBox[video] = []
		duplAct[video] = []
		gtSplitFrames = splitFrames(gtDict[video])
		for f in gtSplitFrames.keys():
			for [idA,typeA,x1A,y1A,x2A,y2A] in gtSplitFrames[f]:
				totalBox += 1
				if idA not in ids:
					ids.append(idA)
					totalAct += 1
				for [idB,typeB,x1B,y1B,x2B,y2B] in gtSplitFrames[f]:
					if idA<idB and [typeA,x1A,y1A,x2A,y2A]==[typeB,x1B,y1B,x2B,y2B]:
						duplBox[video].append([typeA,idA,idB,f])
						hitsBox += 1
						if [idA,idB] not in duplAct[video]:
							duplAct[video].append([idA,idB])
							hitsAct += 1
	duplDict = {}
	for video in sorted(duplBox.keys()):
		idDict = {}
		duplDict[video] = ''
		for actType,idA,idB,f in sorted(duplBox[video]):
			key = actType+'='+str(idA)+'+'+str(idB)
			if key not in idDict.keys():
				idDict[key] = []
			idDict[key].append(f)
		for key in idDict.keys():
			duplDict[video] = duplDict[video]+', '+key+' '+str(min(idDict[key]))+'--'+str(max(idDict[key]))
	with open(outFilePath, 'w') as outFile:
		for video in videos:
			outFile.write(video)
			for dupl in duplDict[video]:
				outFile.write(dupl)
			outFile.write('\n')
		outText = 'Act: '+str(hitsAct)+'/'+str(totalAct)+'='+str(round(100*hitsAct/totalAct,2))+'%\nBox: '+str(hitsBox)+'/'+str(totalBox)+'='+str(round(100*hitsBox/totalBox,2))+'%'
		outFile.write(outText)
		print(outText)

def scanNB(videos, jsonFileName):
	with open(jsonFileName, 'r') as jsonFile:
		j = json.load(jsonFile)
	jDict = {}
	for video in videos:
		jDict[video] = []
		videoName = video+'.mp4'
		for act in j['activities']:
			if videoName in act['localization'].keys():
				actID = act['activityID']
				actType = act['activity']
				beg,end = sorted(list(map(int,act['localization'][videoName].keys())))
				dbb = {'x1':math.inf, 'y1':math.inf, 'x2':-math.inf, 'y2':-math.inf}
				bb = {beg: dbb}
				if 'objects' not in act.keys():
					print('BEWARE: '+'objects'+' not in '+str(act.keys()))
				for obj in act['objects']:
					if str(beg) in obj['localization'][videoName].keys():
						boundingBox = obj['localization'][videoName][str(beg)]['boundingBox']
						x1 = min(bb[beg]['x1'], int(boundingBox['x']))
						y1 = min(bb[beg]['y1'], int(boundingBox['y']))
						x2 = max(bb[beg]['x2'], int(boundingBox['x'])+int(boundingBox['w']))
						y2 = max(bb[beg]['y2'], int(boundingBox['y'])+int(boundingBox['h']))
						bb = {beg: {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}}
				for f in range(beg+1,end):
					bb[f] = dbb
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
							bb[f] = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
				for f in range(beg,end):
					jDict[video].append([actID,actType,f,bb[f]['x1'],bb[f]['y1'],bb[f]['x2'],bb[f]['y2']])
	return jDict

if __name__ == '__main__':
	args = parse_args()
	videoListPath = args.videoListPath
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	detector = args.detector
	tracker = args.tracker
	regressor = args.regressor
	filterer = args.filterer
	detDesc = detector
	trkDesc = detector+'+'+tracker
	act1Desc = detector+'+'+tracker+'+'+regressor
	act2Desc = detector+'+'+tracker+'+'+regressor+'+'+filterer
	varDict = loadINI(args.config, args.index)
	dataFolder = varDict['dataFolder']
	videosFolder = varDict['videosFolder']
	detectionsFolder = os.path.join(varDict['detectionsFolder'], detDesc)
	tracksFolder = os.path.join(varDict['tracksFolder'], trkDesc)
	activities1Folder = os.path.join(varDict['activitiesFolder'], act1Desc)
	activities2Folder = os.path.join(varDict['activitiesFolder'], act2Desc)
	gtFolders = [varDict['gtFolderTrain'], varDict['gtFolderValid']]
	objects = varDict['objects'].split(',')
	activities = varDict['activities'].split(',')
	green = (0,255,0)
	red = (0,0,255)
	blue = (255,0,0)
	purple = (255,0,255)
	yellow = (0,255,255)
	dicList = [None,None,None,None,None]
	abbList = ['','','','','']

	print('Scanning GT'); abbList[0] = 'gt'; gt = gtScan(videos, gtFolders); dicList[0] = gt, green
	#print('Scanning DET'); abbList[1] = 'det'; det = detScan(videos, detectionsFolder, objects); dicList[1] = det, red
	#print('Scanning TRK'); abbList[2] = 'trk'; trk = trkScan(videos, tracksFolder, objects); dicList[2] = trk, blue
	#print('Scanning ACT1'); abbList[3] = 'act1'; act1 = actScan(videos, activities1Folder, activities); dicList[3] = act1, purple
	#print('Scanning ACT2'); abbList[4] = 'act2'; act2 = actScan(videos, activities2Folder, activities); dicList[4] = act2, yellow

	dicList = [dic for dic in dicList if dic is not None]
	abbDesc = '['+','.join([abb for abb in abbList if abb!=''])+']'
	print('Visualizing '+abbDesc.upper()+' on XY'); visXY(videosFolder, os.path.join(dataFolder, 'vis-xy-'+abbDesc), dicList)
	#print('Visualizing '+abbDesc.upper()+' on XT'); visXT(videosFolder, os.path.join(dataFolder, 'vis-xy-'+abbDesc), dicList)

	#fMeasure(gt,trk)

	#print('Duplicate GT'); duplicate(gt, 'duplicate.txt')
	