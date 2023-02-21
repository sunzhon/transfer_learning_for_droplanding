#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    relative_path=$1
else
    relative_path='augmentation_v14'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

collect_test_file="${tflearning_path}/collect_test_files.sh"

results_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
for train_sub_num in $(seq 1 15); do
    for trial_num in $(seq 1 25); do
            #relative_path="augmentation_v6_${config_id}"
            file_dir="$results_path/${relative_path}/${train_sub_num}sub/${trial_num}tri"
            ${collect_test_file} ${file_dir} ${filter_target}
    done
done



