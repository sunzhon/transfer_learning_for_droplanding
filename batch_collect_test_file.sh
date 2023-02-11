#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    relative_path=$1
else
    relative_path='augmentation_v6'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi


results_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
for trial_num in 25; do
    sub_num=15
    for config_id in $(seq 1 10); do
        relative_path="augmentation_v6_${config_id}"
        file_dir="$results_path/${relative_path}/${trial_num}trials/${sub_num}sub"
        ./tf_collect_test_files.sh ${file_dir} ${filter_target}
    done
done



