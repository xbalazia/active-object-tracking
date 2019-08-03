'''
MOSSE tracking sample implements correlation-based tracking approach described in [1]. Useful for tracking target selection. Python 2/3 compatibility.
[1] David S. Bolme et al.: "Visual Object Tracking using Adaptive Correlation Filters"
	http://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
'''
from itertools import count as itercount
from video import create_capture
import numpy as np
import cv2 as cv
import sys, os, argparse, configparser, operator, pickle, datetime


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
	parser.add_argument('--video_list_path', dest='video_list_path', help='video list path', default='virat.txt')
	return parser.parse_args()	

def load_plates(detDictFilePath, trkConfThresh, bbDetMaxArea):
	plates = {}
	if os.path.exists(detDictFilePath):
		with open(detDictFilePath, 'rb') as detDictFile:
			detDict = pickle.load(detDictFile)
		for detID in detDict.keys():
			for f,objType,x1,y1,w,h,conf in detDict[detID]:
				confThreshCheck = float(conf)>=trkConfThresh
				bbDetMaxAreaCheck = w*h<=bbDetMaxArea
				personSizeCheck = not (objType in ['person'] and w>h)
				vehicleSizeCheck = not (objType in ['car','bus'] and w<h)
				if confThreshCheck and bbDetMaxAreaCheck and personSizeCheck and vehicleSizeCheck:
					if f not in plates.keys():
						plates[f] = []
					plates[f].append((x1,y1,w,h,objType))
	else:
		print(detDictFilePath+' does not exist.')
	return plates

def rnd_warp(a):
	h,w = a.shape[:2]
	T = np.zeros((2,3))
	coef = 0.2
	ang = (np.random.rand()-0.5)*coef
	c,s = np.cos(ang), np.sin(ang)
	T[:2,:2] = [[c,-s],[s,c]]
	T[:2,:2] += (np.random.rand(2,2)-0.5)*coef
	c = (w/2,h/2)
	T[:,2] = c-np.dot(T[:2,:2], c)
	return cv.warpAffine(a, T, (w,h), borderMode=cv.BORDER_REFLECT)

def divSpec(A, B):
	Ar, Ai = A[...,0], A[...,1]
	Br, Bi = B[...,0], B[...,1]
	C = (Ar+1j*Ai)/(Br+1j*Bi)
	C = np.dstack([np.real(C), np.imag(C)]).copy()
	return C

def overlap(boxA, boxB):
	x1A,y1A,x2A,y2A = boxA
	x1B,y1B,x2B,y2B = boxB
	return x1A<x1B and x2B<x2A and y1A<y1B and y2B<y2A

def diffPos(boxA, boxB):
	x1A,y1A,x2A,y2A = boxA
	x1B,y1B,x2B,y2B = boxB
	return abs((x1A+x2A)-(x1B+x2B))+abs((y1A+y2A)-(y1B+y2B))

def diffBox(boxA, boxB):
	x1A,y1A,x2A,y2A = boxA
	x1B,y1B,x2B,y2B = boxB
	return abs(x1A-x1B)+abs(y1A-y1B)+abs(x2A-x2B)+abs(y2A-y2B)

class MOSSE:
	def __init__(self, frame, box, trackerID, objType):
		x1,y1,x2,y2 = box
		self.size = w,h = x2-x1,y2-y1
		self.pos = int(x1+0.5*w),int(y1+0.5*h)
		self.trackerID = trackerID
		self.objType = objType
		self.inactiveFrames = 0
		self.psrAvg = 0
		self.updateSize(self.size, frame)

	def updateSize(self, size, frame):
		w,h = size
		if w<bbTrkMinEdge:
			w = bbTrkMinEdge
		if h<bbTrkMinEdge:
			h = bbTrkMinEdge
		self.size = w,h
		img = cv.getRectSubPix(frame, self.size, self.pos)
		self.win = cv.createHanningWindow(self.size, cv.CV_32F)
		g = np.zeros((h,w), np.float32)
		g[int(0.5*h),int(0.5*w)] = 1
		g = cv.GaussianBlur(g, (-1,-1), 2.0)
		g /= g.max()
		self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
		self.H1 = np.zeros_like(self.G)
		self.H2 = np.zeros_like(self.G)
		for _i in xrange(updateSizeIterations):
			a = self.preprocess(rnd_warp(img))
			A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
			self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)
			self.H2 += cv.mulSpectrums(A, A, 0, conjB=True)
		self.update_kernel()
		self.update(frame)
		
	def update(self, frame, rate=0.125):
		(x0,y0), (w,h) = self.pos, self.size
		xMin,yMin = 0,0
		xMax,yMax = frame.shape
		self.last_img = img = cv.getRectSubPix(frame, (w,h), (x0,y0))
		img = self.preprocess(img)
		self.last_resp, (dx,dy), self.psr = self.correlate(img)
		self.pos = x0+dx,y0+dy
		if self.psr>=psrMin and self.psr<=psrMax and x0-0.5*w>=xMin and y0-0.5*h>=yMin and x0+0.5*w<=xMax and y0+0.5*h<=yMax:
			self.inactiveFrames = 0
		else:
			self.inactiveFrames += 1			
		self.psrAvg = self.psrAvg+0.05*(self.psr-self.psrAvg)
		self.last_img = img = cv.getRectSubPix(frame, (w,h), self.pos)
		img = self.preprocess(img)
		A = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
		H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
		H2 = cv.mulSpectrums(A, A, 0, conjB=True)
		self.H1 = self.H1*(1.0-rate)+H1*rate
		self.H2 = self.H2*(1.0-rate)+H2*rate
		self.update_kernel()

	def preprocess(self, img):
		img = np.log(np.float32(img)+1.0)
		img = (img-img.mean())/(img.std()+eps)
		return img*self.win

	def correlate(self, img):
		C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
		resp = cv.idft(C, flags=cv.DFT_SCALE|cv.DFT_REAL_OUTPUT)
		h,w = resp.shape
		_, mval, _, (mx,my) = cv.minMaxLoc(resp)
		side_resp = resp.copy()
		cv.rectangle(side_resp, (mx-5,my-5), (mx+5,my+5), 0, -1)
		smean, sstd = side_resp.mean(), side_resp.std()
		psr = (mval-smean)/(sstd+eps)
		return resp, (int(mx-0.5*w),int(my-0.5*h)), psr

	def update_kernel(self):
		self.H = divSpec(self.H1, self.H2)
		self.H[...,1] *= -1

class App:
	def __init__(self, paused=True):
		self.cap = create_capture(videoPath)
		self.plates = load_plates(os.path.join(detectionsFolder, detector, video+'.pickle'), trkConfThresh, bbDetMaxArea)
		self.trackers = {}
		self.paused = paused
		self.reverseTrackers = {}

	def checkExistTracker(self, boxN):
		currTrackers = self.trackers
		keyMatch = None
		for key, tracker in currTrackers.items():
			(x0T,y0T), (wT,hT) = tracker.pos, tracker.size
			boxT = x0T+0.5*wT,y0T+0.5*hT,x0T-0.5*wT,y0T-0.5*hT
			if overlap(boxN,boxT):
				if not keyMatch:
					keyMatch = key
				else:
					trackerMatch = currTrackers[keyMatch]
					(x0M,y0M), (wM,hM) = trackerMatch.pos, trackerMatch.size
					boxM = x0M+0.5*wM,y0M+0.5*hM,x0M-0.5*wM,y0M-0.5*hM
					if diffPos(boxN,boxT)<diffPos(boxN,boxM):
						keyMatch = key
		return keyMatch
	
	def run(self):
		trkDict = {}
		trackerID = itercount(1)
		begin = datetime.datetime.now()
		f = 0
		while True:
			f += 1
			try:
				ok, self.frame = self.cap.read()
			except:
				print("Error reading video (Frame-%d)..."%f)
			if not ok:
				break
			time = datetime.datetime.now() - begin
			sys.stdout.write('\r'+str(time).split('.')[0]+' '+video+': '+str(f)+fill)
			sys.stdout.flush()
			frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
			detected = []
			if f in self.plates.keys():
				for x1N,y1N,wN,hN,objTypeN in self.plates[f]:
					boxN = (x1N,y1N,x1N+wN,y1N+hN)
					trackerKey = self.checkExistTracker(boxN)
					if f==1 or not trackerKey:
						tracker = MOSSE(frame_gray, boxN, next(trackerID), objTypeN)
						self.trackers[boxN] = tracker
						if f>1:
							self.reverseTrackers[boxN] = (f, tracker)
						detected.append(trackerID)
					else:
						tracker = self.trackers[trackerKey]
						hasMinDiff = True
						(x0T,y0T), (wT,hT) = tracker.pos, tracker.size
						boxT = x0T+0.5*wT,y0T+0.5*hT,x0T-0.5*wT,y0T-0.5*hT
						diffPosNT = diffPos(boxN,boxT)
						if diffPosNT>maxDiffPos:
							continue
						for x1B,y1B,wB,hB,objTypeB in self.plates[f]:
							boxB = (x1B,y1B,x1B+wB,y1B+hB)
							if diffPos(boxB,boxT)<diffPosNT:
								hasMinDiff = False
								break
						if hasMinDiff:
							tracker.pos = (int(x1N+0.5*wN),int(y1N+0.5*hN))
							tracker.updateSize((wN,hN), frame_gray)
							tracker.objType = objTypeN
							tracker.inactiveFrames = 0
							detected.append(tracker.trackerID)
			for trackerKey in list(self.trackers):
				tracker = self.trackers[trackerKey]
				trkID = tracker.trackerID
				if trkID not in detected:
					tracker.update(frame_gray)
					tracker.objType = None
				objType = tracker.objType
				(x0,y0), (w,h) = tracker.pos, tracker.size
				if trkID not in trkDict.keys():
					trkDict[trkID] = {}
				trkDict[trkID][f] = objType,(int(x0-0.5*w),int(y0-0.5*h),w,h)
				if tracker.inactiveFrames>maxInactiveFrames:
					del self.trackers[trackerKey]
		sys.stdout.write('\n')
		delIDs = []
		for trkID in trkDict.keys():
			objTypeCountDict = {}
			for f in trkDict[trkID].keys():
				objType,_ = trkDict[trkID][f]
				if objType!=None:
					if objType not in objTypeCountDict.keys():
						objTypeCountDict[objType] = 0
					objTypeCountDict[objType] += 1
			if objTypeCountDict:
				objTypeFinal = max(objTypeCountDict.items(), key=operator.itemgetter(1))[0]
				if objTypeFinal in objects and objTypeCountDict[objTypeFinal]>=minDets:
					bbDict = {}
					for f in trkDict[trkID].keys():
						_,bbDict[f] = trkDict[trkID][f]
					trkDict[trkID] = objTypeFinal, bbDict
				else:
					delIDs.append(trkID)
			else:
				delIDs.append(trkID)
		for delID in delIDs:
			del trkDict[delID]
		self.cap.release()
		cv.destroyAllWindows()
		if not os.path.exists(trkFolder):
			os.makedirs(trkFolder)
		with open(os.path.join(trkFolder, video+'.pickle'), 'wb') as trkDictFile:
			pickle.dump(trkDict, trkDictFile)

if __name__ == '__main__':
	args = parse_args()
	video_list_path = args.video_list_path
	varDict = loadINI(args.config, args.index)
	videosFolder = varDict['videosFolder']
	detectionsFolder = varDict['detectionsFolder']
	detector = varDict['detector']
	trkConfThresh = float(varDict['trkConfThresh'])
	tracker = 'mosse'
	trkFolder = os.path.join(varDict['tracksFolder'], detector+'+'+tracker)
	bbDetMaxArea = 60000 # 'maximum area of detection bounding box'
	bbTrkMinEdge = 15 # 'minimum edge of tracking bounding box'
	maxDiffPos = 65 # 'maximum difference between detection and tracking bounding box'
	psrMin = 7 # 'minimum PSR value'
	psrMax = 50 # 'maximum PSR value'
	minDets = 2 # 'minimum number of detections in track'
	updateSizeIterations = 16 # 'number of updateSize iterations (originally 128)'
	maxInactiveFrames = 20 # 'number of inactive frames before tracker disappears'
	objects = varDict['objects'].split(',')
	eps = 1e-5
	PY3 = sys.version_info[0]==3
	if PY3:
		xrange = range
	fill = '        '
	with open(args.video_list_path,'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	for video in videos:
		videoPath = os.path.join(videosFolder, video+'.mp4')
		App().run()