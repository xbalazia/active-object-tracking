#!/bin/bash

declare -r gpu=$1

declare -a activities=("Closing_Trunk" "Closing" "Entering" "Exiting" "Loading" "Open_Trunk" "Opening" "Pull" "Riding" "Talking" "Transport_HeavyCarry" "Unloading" "activity_carrying" "specialized_talking_phone" "specialized_texting_phone" "vehicle_turning_left" "vehicle_turning_right" "vehicle_u_turn")

for activity in "${activities[@]}"; do
	echo "python3 lstm/training.py -g $gpu -a $activity"
	python3 lstm/training.py -g $gpu -a $activity
done