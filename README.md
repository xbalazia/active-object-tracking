# Active Object Tracking

# SSD Detector:
# Code was originally obtained from
# https://github.com/elranu/ssd_pi/blob/master/ssd_predictor.py
# with pre-trained weights from
# https://github.com/elranu/ssd_pi/blob/master/trained_weights/VGG_coco_SSD_512x512.h5
# Run detection with command
# $ sh run-ssd-detection.sh ssd videos-valid.txt 0

# MOSSE Tracker:
# Code was originally obtained from
# https://github.com/opencv/opencv/blob/master/samples/python/mosse.py
# Run tracking with command
# $ sh run-mosse-tracking.sh videos-valid.txt

# VGG16+LSTM+HCF Regressor:
# Run training with command
# $ sh run-lstm-training.sh 0
# Run inference with command
# $ sh run-lstm-inference.sh videos-valid.txt 0
