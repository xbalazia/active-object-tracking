import argparse, configparser, pickle

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
	outFilePath = 'duplicates.txt'
	args = parse_args()
	videoListPath = args.videoListPath
	varDict = loadINI(args.config, args.index)
	gtDictFolder = varDict['gtDictFolder']
	hitsBox = 0
	hitsAct = 0
	totalBox = 0
	totalAct = 0
	duplBox = {}
	duplAct = {}
	ids = []
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	for video in videos:
		duplBox[video] = []
		duplAct[video] = []
		frames = {}
		with open(gtDictFolder+'/'+video+'.pickle', 'rb') as gtDictFile:
			gtDict = pickle.load(gtDictFile)
		for gtID in gtDict.keys():
			bbDict, _, _, act, _ = gtDict[gtID]
			for f in bbDict.keys():
				x1,y1,w,h = bbDict[f]
				if f not in frames.keys():
					frames[f] = []
				frames[f].append([gtID,act,x1,y1,x1+w,y1+h])
		for f in frames.keys():
			for [idA,typeA,x1A,y1A,x2A,y2A] in frames[f]:
				totalBox += 1
				if idA not in ids:
					ids.append(idA)
					totalAct += 1
				for [idB,typeB,x1B,y1B,x2B,y2B] in frames[f]:
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
		outText = 'Found duplicate GT annotations in '+str(hitsAct)+'/'+str(totalAct)+'='+str(round(100*hitsAct/totalAct,2))+'% '+'activities or '+str(hitsBox)+'/'+str(totalBox)+'='+str(round(100*hitsBox/totalBox,2))+'% '+'bounding boxes'
		outFile.write(outText)
		print(outText)