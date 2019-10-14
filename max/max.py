import os, argparse, configparser, json, pickle

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
	return parser.parse_args()

# Main
if __name__ == '__main__':

	# Load config and gets variables
	args = parse_args()
	videoListPath = args.videoListPath
	varDict = loadINI(args.config, args.index)
	detector = varDict['detector']
	tracker = varDict['tracker']
	regressor = varDict['regressor']+'_'+args.model
	filterer = varDict['filterer']
	inDesc = detector+'+'+tracker+'+'+regressor+'+'+filterer
	outDesc = detector+'+'+tracker+'+'+regressor+'+'+filterer+'+max'
	inFolder = os.path.join(varDict['activitiesFolder'], inDesc)
	outFolder = os.path.join(varDict['activitiesFolder'], outDesc)

	filesProcessed = []
	actList = []
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	for video in videos:
		with open(os.path.join(inFolder, video+'.pickle'), 'rb') as actDictPickleFile:
			actDict = pickle.load(actDictPickleFile)
		maxDict = {}
		for actID in actDict.keys():
			bbDict, objType, trkID, activity, conf = actDict[actID]
			for f in bbDict.keys():
				key = 1000000*trkID+f
				if key not in maxDict.keys():
					maxDict[key] = {f: bbDict[f]}, objType, trkID, 'active', conf
				else:
					bbDict_O, objType_O, trkID_O, activity_O, conf_O = maxDict[key]
					if conf_O<conf:
						maxDict[key] = bbDict_O, objType_O, trkID_O, activity_O, conf

		# Create video output: CSV, JSON, PICKLE
		if not os.path.exists(outFolder):
			os.makedirs(outFolder)
		with open(os.path.join(outFolder, video+'.pickle'), 'wb') as maxDictFile:
			pickle.dump(maxDict, maxDictFile)
		with open(os.path.join(outFolder, video+'.csv'), 'w') as csvFile, open(os.path.join(outFolder, video+'.json'), 'w') as jsonFile:
			filesProcessed.append(video+'.mp4')
			jsonData = {u'filesProcessed': [video+'.mp4'], u'activities':[]}
			for key in maxDict.keys():
				bbDict, objType, trkID, activity, conf = maxDict[key]
				beginFrame, endFrame = min(bbDict.keys()), max(bbDict.keys())
				tube = {}
				for f in bbDict.keys():
					x1,y1,w,h = bbDict[f]
					tube[str(f)] = {'boundingBox': {'h': h, 'w': w, 'x': x1, 'y': y1}}
				csvFile.write(','.join(str(v) for v in [key, objType, trkID, beginFrame, endFrame, activity, conf])+'\n')
				jsonDict = {
					'activity': activity,
					'activityID': key,
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
				actList.append(jsonDict)
			json.dump(jsonData, jsonFile, indent=2)