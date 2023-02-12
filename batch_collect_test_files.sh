#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    relative_path=$1
else
    relative_path='baseline_v9_5'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

collect_test_ile="${tflearning_path}/collect_test_files.sh"

results_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
for trial_num in 25; do
    for sub_num in $(seq 2 15); do
        #relative_path="augmentation_v6_${config_id}"
        file_dir="$results_path/${relative_path}/${trial_num}trials/${sub_num}sub"
        ${collect_test_ile} ${file_dir} ${filter_target}
    done
done



