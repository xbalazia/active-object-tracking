# Active Object Tracking

Download the VIRAT dataset from
http://www.viratdata.org/
and take ground truth annotations from the gt.rar archive.

## Extract Ground Truth
Extract GT dictionaries with command
```
$ python3 gt-extract-dict.py -v videos-gt.txt
```

## Find Duplicate Annotations in Ground Truth
Find GT duplicates with command
```
$ python3 gt-find-duplicates.py -v videos-gt.txt
```

## SSD Detector
Code was originally obtained from
https://github.com/elranu/ssd_pi/blob/master/ssd_predictor.py
with pre-trained weights from
https://github.com/elranu/ssd_pi/blob/master/trained_weights/VGG_coco_SSD_512x512.h5

Run detection with command
```
$ python3 ssd/ssd.py -v videos-valid.txt
```

## MOSSE Tracker
Code was originally obtained from
https://github.com/opencv/opencv/blob/master/samples/python/mosse.py

Run tracking with command
```
$ python3 mosse/mosse.py videos-valid.txt
```

## VGG16+LSTM+HCF+NMS Regressor
Run training with command
```
$ ./run-training.sh
```
Run inference with command
```
$ ./run-inference.sh videos-valid.txt
```

## Evaluate
Evaluate regression with command
```
$ python3 eval/eval.py -v vids-valid.txt
```
