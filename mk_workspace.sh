#! /bin/zsh

if [ $# -gt 0 ]; then
    dir_path=$1
else
    dir_path=${HOME}
fi

echo $dir_path
#1) make folders of drop_landing workspace, dataset folder and result folder
result_folder="$dir_path/drop_landing_workspace/results/training_testing"
dataset_folder="$dir_path/drop_landing_workspace/"

if [[ ! -d $result_folder ]]; then
    mkdir -p $result_folder
fi

if [[ ! -d $dataset_folder ]]; then
    mkdir -p $dataset_folder
fi

#2) copy dataset to drop landing workspace dataset folder
remote_rawdataset_folder="/mnt/drop_landing_workspace/data/"
local_1_rawdataset_folder="/media/sun/DATA/drop_landing_workspace/data/"

if [[ -d $remote_rawdataset_folder ]]; then
    cp -r "/mnt/drop_landing_workspace/data/" $dataset_folder
elif [[ -d $local_1_rawdataset_folder ]]; then
    cp -r "/media/sun/DATA/drop_landing_workspace/data/" $dataset_folder
else
    cp -r "/media/sun/My\ Passport/drop_landing_workspace/data/" $dataset_folder
fi

export MEDIA_NAME="$dir_path"
