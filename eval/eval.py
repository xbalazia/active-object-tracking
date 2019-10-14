import os, argparse, configparser, math, json, pickle, csv
import numpy as np
import matplotlib.pyplot as plt

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
	parser.add_argument('-t', '--types', dest='types', default='det,trk,act1,act2,act3', help='Types to evaluate.')
	return parser.parse_args()

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

def	getHitAndTotalCounts(dict1, dict2, decks):
	hitCounts = []
	totalCounts = []
	for i in range(decks):
		hitCount = 0
		totalCount = 0
		for id1 in dict1.keys():
			bbDict1, _, _, _, conf1 = dict1[id1]
			for f in bbDict1.keys():
				if conf1>=i/decks:
					totalCount += 1
					for id2 in dict2.keys():
						bbDict2, _, _, _, conf2 = dict2[id2]
						if f not in bbDict2.keys():
							continue
						if isOverlapping(bbDict1[f], bbDict2[f]):
							if conf2>=i/decks:
								hitCount += 1
							break
		hitCounts.append(hitCount)
		totalCounts.append(totalCount)
	return hitCounts, totalCounts

def	getRate(hitsCount, totalCount, decks):
	rate = []
	for i in range(decks):
		if totalCount[i]==0:
			rate.append(0)
		else:
			rate.append(1-hitsCount[i]/totalCount[i])
	return rate

# Main
if __name__ == '__main__':

	# Load config and get variables
	args = parse_args()
	videoListPath = args.videoListPath
	varDict = loadINI(args.config, args.index)
	gtDictFolder = varDict['gtDictFolder']
	detFolder = varDict['detectionsFolder']
	trkFolder = varDict['tracksFolder']
	actFolder = varDict['activitiesFolder']
	evalFolder = varDict['evaluationsFolder']
	detector = varDict['detector']
	tracker = varDict['tracker']
	regressor = varDict['regressor']+'_'+args.model
	filterer = varDict['filterer']
	detDesc = detector
	trkDesc = detector+'+'+tracker
	act1Desc = detector+'+'+tracker+'+'+regressor
	act2Desc = detector+'+'+tracker+'+'+regressor+'+'+filterer
	act3Desc = detector+'+'+tracker+'+'+regressor+'+'+filterer+'+max'
	red = (1,0,0)
	blue = (0,0,1)
	purple = (1,0,1)
	yellow = (1,1,0)
	cyan = (0,1,1)
	types = args.types.split(',')
	typeDict =	{	'det':	[os.path.join(detFolder, detDesc),	'SSD', 						red], 		#detDesc	#'SSD'
					'trk':	[os.path.join(trkFolder, trkDesc),	'SSD+MOSSE',				blue],		#trkDesc	#'SSD+MOSSE'
					'act1':	[os.path.join(actFolder, act1Desc),	'SSD+MOSSE+LSTM',			purple],	#act1Desc	#'SSD+MOSSE+LSTM'
					'act2':	[os.path.join(actFolder, act2Desc),	'SSD+MOSSE+LSTM+HCF',		yellow],	#act2Desc	#'SSD+MOSSE+LSTM+HCF'
					'act3':	[os.path.join(actFolder, act3Desc),	'SSD+MOSSE+LSTM+HCF+NMS',	cyan],		#act3Desc	#'SSD+MOSSE+LSTM+HCF+NMS'
				}
	decks = 10
	descs = ','.join(typeDict[typ][1] for typ in types)
	evalFilePath = os.path.join(evalFolder, 'DET['+videoListPath+']['+descs+'].png')

	# Run evaluations
	for typ in types:
		infDictFolder, infDesc, infColor = typeDict[typ]
		gtHitCounts = []
		gtTotalCounts = []
		infHitCounts = []
		infTotalCounts = []
		with open(videoListPath, 'r') as vidListFile:
			videos = vidListFile.read().splitlines()
		for video in videos:
			print('Evaluating', video, 'with', infDesc)
			gtDictFilePath = os.path.join(gtDictFolder, video+'.pickle')
			infDictFilePath = os.path.join(infDictFolder, video+'.pickle')
			if not os.path.exists(gtDictFilePath):
				print(gtDictFilePath, 'does not exist.')
				continue
			if not os.path.exists(infDictFilePath):
				print(infDictFilePath, 'does not exist.')
				continue
			with open(gtDictFilePath, 'rb') as gtDictFile:
				gtDict = pickle.load(gtDictFile)
			with open(infDictFilePath, 'rb') as infDictFile:
				infDict = pickle.load(infDictFile)
			gtHitCountsVideo, gtTotalCountsVideo = getHitAndTotalCounts(gtDict, infDict, decks)
			infHitCountsVideo, infTotalCountsVideo = getHitAndTotalCounts(infDict, gtDict, decks)
			gtHitCounts += gtHitCountsVideo
			gtTotalCounts += gtTotalCountsVideo
			infHitCounts += infHitCountsVideo
			infTotalCounts += infTotalCountsVideo
		pmd = getRate(gtHitCounts, gtTotalCounts, decks)
		rfa = getRate(infHitCounts, infTotalCounts, decks)
		mixDict = {}
		for i in range(len(rfa)):
			if rfa[i] in mixDict.keys():
				mixDict[rfa[i]] = min(mixDict[rfa[i]], pmd[i])
			else:
				mixDict[rfa[i]] = pmd[i]
		keys = sorted(mixDict.keys())
		for x1 in keys:
			if x1 not in mixDict.keys():
				continue
			for x2 in keys:
				if x2 not in mixDict.keys():
					continue
				if x1<x2 and mixDict[x1]<=mixDict[x2]:
					del mixDict[x2]
		keys = sorted(mixDict.keys())
		plotX = []
		plotY = []
		for i in range(len(keys)):
			if i==0:
				plotX.append(0)
			else:
				plotX.append(keys[i-1])
			plotY.append(mixDict[keys[i]])
			plotX.append(keys[i])
			plotY.append(mixDict[keys[i]])
		plt.plot(plotX, plotY, label=infDesc, color=infColor)

	# Plot
	plt.title('Detection Error Tradeoff')
	plt.xlabel('Rate of False Alarm')
	plt.ylabel('Probability of Misdetection')
	plt.axis([0, 1, 0, 1])
	plt.xticks(np.arange(0,1.01,0.1))
	plt.yticks(np.arange(0,1.01,0.1))
	plt.grid(True)
	plt.legend(fontsize='small')
	plt.savefig(evalFilePath)
	print('Saved', evalFilePath)