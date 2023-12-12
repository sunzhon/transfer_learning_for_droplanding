#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

for tmp in $@; do 
	./investigation.sh $tmp 5
done;

