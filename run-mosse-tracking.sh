#!/usr/bin/env bash

declare -r trk=$1
declare -r vlp=$2

python $trk \
	--video_list_path $vlp