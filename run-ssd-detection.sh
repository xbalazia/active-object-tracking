#!/usr/bin/env bash

declare -r det=$1
declare -r vlp=$2
declare -r gpu=$3

python3 $det \
	--video_list_path $vlp \
	--gpu $gpu \