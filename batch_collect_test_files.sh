#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    result_folder=$1
else
    result_folder='rdouble_leg_v1_baseline_4_original_7_15_25_R_KNEE_MOMENT_X_ori_v2'
fi

# read test_result folder from a txt file
#dir_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
#result_folder_array=($(cat "${dir_path}/all_test_cases.txt"))


#for result_folder in ${result_folder_array[@]}; do
    echo ${result_folder}


    if [ $# -gt 1 ]; then
        filter_target=$2
    else
        filter_target='test_'
    fi

    collect_sh="${tflearning_path}/collect_test_files.sh"

    dir_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
    file_dir="${dir_path}/${result_folder}"
    ${collect_sh} ${file_dir} ${filter_target}

#done
