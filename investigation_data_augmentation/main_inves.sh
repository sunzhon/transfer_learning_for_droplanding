#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

#for sub in $(seq 5 14); do 
	# investigation.sh: $1 is subject num, $2 is tre trial num
	for trial in $(seq 1 5 25); do
		echo "sub: $sub, trial: $trial"
		#./investigation.sh $sub $trial
		./investigation.sh $trial
	done;
#done;

