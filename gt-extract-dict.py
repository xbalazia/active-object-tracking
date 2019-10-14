import os, argparse, configparser, math, json, pickle

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
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	videoListPath = args.videoListPath
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	varDict = loadINI(args.config, args.index)
	gtFolders = [varDict['trainFolder'], varDict['validFolder']]
	gtDictFolder = varDict['gtDictFolder']
	for video in videos:
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
		gtDict = {}
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
			bbDict = {}
			for f in range(beg,end):
				bbDict[f] = [bb[f]['x1'],bb[f]['y1'],bb[f]['x2']-bb[f]['x1'],bb[f]['y2']-bb[f]['y1']]
			#bbDict, objType, trkID, act, conf
			gtDict[actID] = [bbDict, '', 0, actType, 1]
		if not os.path.exists(gtDictFolder):
			os.makedirs(gtDictFolder)
		with open(os.path.join(gtDictFolder, video+'.pickle'), 'wb') as gtDictFile:
			pickle.dump(gtDict, gtDictFile)