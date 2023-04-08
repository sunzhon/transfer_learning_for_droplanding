#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    result_folder=$1
else
    result_folder='augmentation_dkem_v5/6rotid'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

collect_sh="${tflearning_path}/collect_test_files.sh"

dir_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
file_dir="${dir_path}/${result_folder}"
${collect_sh} ${file_dir} ${filter_target}



