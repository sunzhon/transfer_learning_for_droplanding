#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8


result_folder="./result_folders.txt" 

tmp_result_folder_array=()
readarray tmp_result_folder_array < ${result_folder} 

result_folder_array=()
for ii in ${tmp_result_folder_array[@]}; do
    result_folder_array+=($(echo $ii | tr '\n' ' '))
done

#echo "a: ${result_folder_array[1]}"
python ./../assessments/scores.py  "${result_folder_array[@]}"
