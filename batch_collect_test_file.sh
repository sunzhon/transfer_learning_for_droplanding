#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    methods=$1
else
    methods='intersub_normal_dann'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi



results_path='/home/sun/drop_landing_workspace/results/training_testing'
for trial_num in 5 10 15 20 25; do
    #for sub_num in $(seq 5 11); do
    for sub_num in '08' '10' '11' '14' 15 16 17 18 19 20 21 22 23 24; do
        #file_dir="$results_path/investigation_${methods}_v1/${trial_num}trials/${sub_num}sub"
        file_dir="$results_path/investigation_${methods}_v1/${sub_num}sub"
        ./tf_collect_test_files.sh ${file_dir} ${filter_target}
    done
done



