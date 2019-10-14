#!/bin/bash

declare -r vlp=$1
declare -r mod=$2
declare -r typ=$3
declare -r gpu=$4

#echo "python3 frcnn/frcnn.py -v $vlp -g $gpu"
#python3 frcnn/frcnn.py -v $vlp -g $gpu

#echo "python3 ssd/ssd.py -v $vlp -g $gpu"
#python3 ssd/ssd.py -v $vlp -g $gpu

#echo "python3 mosse/mosse.py -v $vlp"
#python3 mosse/mosse.py -v $vlp

echo "python3 lstm/inference.py -v $vlp -m $mod -g $gpu"
python3 lstm/inference-binary.py -v $vlp -m $mod -g $gpu

echo "python3 hcf/filter.py -v $vlp -m $mod"
python3 hcf/filter.py -v $vlp -m $mod

echo "python3 max/max.py -v $vlp -m $mod"
python3 max/max.py -v $vlp -m $mod

#echo "python3 eval/eval.py -v $vlp -m $mod -t $typ"
#python3 eval/eval.py -v $vlp -m $mod -t $typ