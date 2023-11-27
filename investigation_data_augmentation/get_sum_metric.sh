#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
# argument
if [ $# -gt 0 ]; then
    result_folder_file=$1
else
    result_folder_file="./result_folders.txt" 
fi

echo "Result folders: ${result_folder_file}"

sleep 2

# getting folders
tmp_result_folder_array=()
readarray tmp_result_folder_array < ${result_folder_file}

# remove \n
result_folder_array=()
for ii in ${tmp_result_folder_array[@]}; do
    result_folder_array+=($(echo $ii | tr '\n' ' '))
done

# collect testing_result_folders.txt

#if [ true ]; then
#for result_folder in ${result_folder_array[@]}; do
#    echo "result folder:  ${result_folder}"
#./../batch_collect_test_files.sh ${result_folder}
#done
#fi

# cclculate metrics
#echo "a: ${result_folder_array[1]}"
python ./../assessments/scores.py  "${result_folder_array[@]}"
