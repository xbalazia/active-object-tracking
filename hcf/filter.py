import os, argparse, configparser, math, json, pickle
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
	parser.add_argument('-v', '--videoListPath', dest='videoListPath', default='vids.txt', help='Path to video list file.')
	return parser.parse_args()

def delete(actDict, actID):
	if actID in actDict.keys():
		del actDict[actID]

# Main
if __name__ == '__main__':

	# Load config and gets variables
	args = parse_args()
	videoListPath = args.videoListPath
	setup = videoListPath.replace('.txt','')
	varDict = loadINI(args.config, args.index)
	detector = varDict['detector']
	tracker = varDict['tracker']
	regressor = varDict['regressor']
	filterer = varDict['filterer']
	inDesc = detector+'+'+tracker+'+'+regressor+'_m1' ###
	outDesc = detector+'+'+tracker+'+'+regressor+'_m1+'+filterer ###
	inFolder = os.path.join(varDict['activitiesFolder'], inDesc)
	outFolder = os.path.join(varDict['activitiesFolder'], outDesc)
	scoresFolder = os.path.join(varDict['scoresFolder'], outDesc)
	
	# HCF filter
	print('HCF filtering '+videoListPath)
	filesProcessed = []
	actList = []
	with open(videoListPath, 'r') as vidListFile:
		videos = vidListFile.read().splitlines()
	for video in videos:
		print('HCF filtering video '+video) ###
		actDictPicklePath = os.path.join(inFolder, video+'.pickle')
		if not os.path.exists(actDictPicklePath):
			print(actDictPicklePath+' does not exist.')
			break
		with open(actDictPicklePath, 'rb') as actDictPickleFile:
			actDict = pickle.load(actDictPickleFile)
		for actID in list(actDict.keys()):
			bbDict, objType, trkID, activity, conf = actDict[actID]
			length = len(bbDict)
			posDict = {}
			x0_a,y0_a = 0,0
			for f in bbDict.keys():
				x1,y1,x2,y2 = bbDict[f]
				x0,y0 = (x1+x2)/2,(y1+y2)/2
				posDict[f] = (x0,y0)
				x0_a,y0_a = x0+x0_a,y0+y0_a
			x0_a,y0_a = x0_a/length,y0_a/length

			# Filter by object type
			# if activity in ['Entering', 'Exiting'] and objType not in ['car', 'bus']:
			# 	delete(actDict, actID); continue
			# if activity in ['Opening', 'Closing'] and objType not in ['car', 'bus']:
			# 	delete(actDict, actID); continue
			# if activity in ['Loading', 'Unloading'] and objType not in ['person']:
			# 	delete(actDict, actID); continue
			# if activity in ['Open_Trunk', 'Closing_Trunk'] and objType not in ['person']:
			# 	delete(actDict, actID); continue

			# Filter by travel index
			travelIndex = 0
			for f in posDict.keys():
				x0,y0 = posDict[f]
				travelIndex += (abs(x0-x0_a)+abs(y0-y0_a))/length
			if activity in ['specialized_talking_phone', 'specialized_texting_phone'] and not 1<travelIndex:
				delete(actDict, actID); continue
			if activity in ['Entering', 'Exiting'] and not 5<travelIndex<10:		#5,10 #3,15
				delete(actDict, actID); continue
			if activity in ['Opening', 'Closing'] and not 1<travelIndex<5:					#1,5 #3,15
				delete(actDict, actID); continue
			if activity in ['Open_Trunk', 'Closing_Trunk'] and not 1<travelIndex<5:		#1,5  #3,15
				delete(actDict, actID); continue
			if activity in ['Loading', 'Unloading'] and not 1<travelIndex<10:		#1,10 #3,15
				delete(actDict, actID); continue
			if activity in ['Talking'] and not 5<travelIndex<10:		#5,10 #3,10
				delete(actDict, actID); continue
			if activity in ['activity_carrying'] and not 40<travelIndex:
				delete(actDict, actID); continue
			if activity in ['Transport_HeavyCarry'] and not 40<travelIndex:		#40 #70
				delete(actDict, actID); continue
			if activity in ['Pull'] and not 40<travelIndex:
				delete(actDict, actID); continue
			if activity in ['vehicle_turning_left', 'vehicle_turning_right'] and not 30<travelIndex:
				delete(actDict, actID); continue
			if activity in ['vehicle_u_turn'] and not 60<travelIndex:
				delete(actDict, actID); continue
			
			# Filter by travel angle
			bx,ex,by,ey = 0,0,0,0
			beginFrame,endFrame = min(bbDict.keys()),max(bbDict.keys())
			p = int(length/4)
			bxp,byp = posDict[beginFrame]
			exp,eyp = posDict[endFrame-p]
			for i in range(1, p):
				bxc,byc = posDict[beginFrame+i]
				exc,eyc = posDict[endFrame-p+i]
				bx,ex,by,ey = bx+(bxc-bxp)/p,ex+(exc-exp)/p,by+(byc-byp)/p,ey+(eyc-eyp)/p
				bxp,exp,byp,eyp = bxc,exc,byc,eyc
			travelAngle = math.atan2(bx*ey-by*ex, bx*ex+by*ey)
			if activity in ['vehicle_turning_left'] and not -2<travelAngle<-0.5:
				delete(actDict, actID); continue
			if activity in ['vehicle_turning_right'] and not 0.5<travelAngle<2:
				delete(actDict, actID); continue
			if activity in ['vehicle_u_turn'] and not 0.5<abs(travelAngle):
				delete(actDict, actID); continue

			# Filter by closest object
			# minDistance = math.inf
			# for actID_c in actDict.keys():
			# 	bbDict_c, objType_c, trkID_c, activity_c, conf_c = actDict[actID_c]
			# 	intersection = set(bbDict.keys())&set(bbDict_c.keys())
			# 	if trkID==trkID_c or len(intersection)==0:
			# 		continue
			# 	for f in intersection:
			# 		x0,y0 = posDict[f]
			# 		x1_c,y1_c,x2_c,y2_c = bbDict_c[f]
			# 		x0_c,y0_c = (x1_c+x2_c)/2,(y1_c+y2_c)/2
			# 		distance_c = abs(x0-x0_c)+abs(y0-y0_c)
			# 		if distance_c<minDistance:
			# 			minDistance = distance_c
			# 			closestObject = objType_c
			# print(video, activity, closestObject)
			# if activity in ['Talking'] and closestObject not in ['person']:
			# 	delete(actDict, actID); continue

		# Unify temporally overlapping duplicate activities
		for actID_1 in list(actDict.keys()):
			if actID_1 not in actDict.keys():
				continue
			bbDict_1, objType_1, trkID_1, activity_1, conf_1 = actDict[actID_1]
			for actID_2 in list(actDict.keys()):
				if actID_1==actID_2 or actID_1 not in actDict.keys() or actID_2 not in actDict.keys():
					continue
				bbDict_2, objType_2, trkID_2, activity_2, conf_2 = actDict[actID_2]
				if conf_1<conf_2:
					actID_A = actID_1
					actID_B = actID_2
				else:
					actID_A = actID_2
					actID_B = actID_1
				bbDict_A, objType_A, trkID_A, activity_A, conf_A = actDict[actID_A]
				bbDict_B, objType_B, trkID_B, activity_B, conf_B = actDict[actID_B]
				if trkID_A==trkID_B and activity_A==activity_B:
					intersection = set(bbDict_A.keys())&set(bbDict_B.keys())
					union = set(bbDict_A.keys())|set(bbDict_B.keys())
					if len(intersection)>3:
						if conf_A/conf_B>0.7:
							bbDict_u = dict(list(bbDict_A.items())+list(bbDict_B.items()))
							actDict[actID_B] = [bbDict_u, objType_B, trkID_B, activity_B, (conf_A+conf_B)/2]
						#actDict[actID_B] = [dict(list(bbDict_A.items())+list(bbDict_B.items())), objType_B, trkID_B, activity_B, (conf_A+conf_B)/2]
						delete(actDict, actID_A)
					# if len(intersection)<=3 and max(union)-min(union)-len(union)<10 and conf_A/conf_B>0.7:
					# 	actDict[actID_B] = [dict(list(bbDict_A.items())+list(bbDict_B.items())), objType_B, trkID_B, activity_B, (conf_A+conf_B)/2]
					# 	delete(actDict, actID_A)

		# Induce superclasses
		for actID in list(actDict.keys()):
			bbDict, objType, trkID, activity, conf = actDict[actID]
			if activity in ['Entering','Exiting']:
				actDict[actID+100000] = [bbDict, objType, trkID, 'Opening', conf]
				actDict[actID+200000] = [bbDict, objType, trkID, 'Closing', conf]
			#if activity in ['Loading','Unloading']:
			#	actDict[actID+300000] = [bbDict, objType, trkID, 'Open_Trunk', conf]
			#	actDict[actID+400000] = [bbDict, objType, trkID, 'Closing_Trunk', conf]
			#if activity in ['Transport_HeavyCarry']:
			#	actDict[actID+500000] = [bbDict, objType, trkID, 'activity_carrying', conf]

		# Non-maximum suppression
		# for actID_1 in list(actDict.keys()):
		# 	if actID_1 not in actDict.keys():
		# 		continue
		# 	bbDict_1, objType_1, trkID_1, activity_1, conf_1 = actDict[actID_1]
		# 	for actID_2 in list(actDict.keys()):
		# 		if actID_1==actID_2 or actID_1 not in actDict.keys() or actID_2 not in actDict.keys():
		# 			continue
		# 		bbDict_2, objType_2, trkID_2, activity_2, conf_2 = actDict[actID_2]
		# 		intersection = set(bbDict_1.keys())&set(bbDict_2.keys())
		# 		if trkID_1==trkID_2 and len(intersection)>20:
		# 			if conf_1<conf_2:
		# 				delete(actDict, actID_1)
		# 			else:
		# 				delete(actDict, actID_2)

		# Create video output: CSV, JSON, PICKLE
		if not os.path.exists(outFolder):
			os.makedirs(outFolder)
		with open(os.path.join(outFolder, video+'.pickle'), 'wb') as actDictFile:
			pickle.dump(actDict, actDictFile)
		with open(os.path.join(outFolder, video+'.csv'), 'w') as csvFile, open(os.path.join(outFolder, video+'.json'), 'w') as jsonFile:
			filesProcessed.append(video+'.mp4')
			jsonData = {u'filesProcessed': [video+'.mp4'], u'activities':[]}
			for actID in actDict.keys():
				bbDict, objType, trkID, activity, conf = actDict[actID]
				beginFrame,endFrame = min(bbDict.keys()),max(bbDict.keys())
				actIDs = list(str(actID))
				actIDs[0] = 'V'
				csvFile.write(','.join(str(v) for v in [''.join(actIDs), objType, trkID, beginFrame, endFrame, activity, conf])+'\n')
				jsonDict = {
					'activity': activity,
					'activityID': actID,
					'presenceConf': float(conf),
					'alertFrame': endFrame,
					'localization': {
						video+'.mp4': {
							beginFrame: 1,
							endFrame: 0
						}
					}
				}
				jsonData[u'activities'].append(jsonDict)
				actList.append(jsonDict)
			json.dump(jsonData, jsonFile, indent=2)

	# Create final output: JSON
	if not os.path.exists(scoresFolder):
		os.makedirs(scoresFolder)
	jsonFinalPath = os.path.join(scoresFolder, setup+'.json')
	with open(jsonFinalPath, 'w') as jsonFinalFile:
		jsonFinalData = {u'filesProcessed': filesProcessed, u'activities': actList}
		json.dump(jsonFinalData, jsonFinalFile, indent=2)
	print('Created '+jsonFinalPath+' of '+str(len(actList))+' activities')
