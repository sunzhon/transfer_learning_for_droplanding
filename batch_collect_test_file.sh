#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    relative_path=$1
else
    relative_path='baseline_v6_1_5_10'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi


results_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
for trial_num in 25; do
    sub_num=15
    #for sub_num in $(seq 5 11); do
    for relative_path in 'baseline_v6_2'; do # 'baseline_v6_4' 'baseline_v6_6' 'baseline_v6_8'  ; do
        file_dir="$results_path/${relative_path}/${trial_num}trials/${sub_num}sub"
        ./tf_collect_test_files.sh ${file_dir} ${filter_target}
    done
done



