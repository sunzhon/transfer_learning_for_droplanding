#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    relative_path=$1
else
    relative_path='augmentation_dkem_v5/6rotid'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

collect_test_file="${tflearning_path}/collect_test_files.sh"

results_path="${MEDIA_NAME}drop_landing_workspace/results/training_testing"
for relative_path in "augmentation_t16"; do # "baseline_mlnn_t16"; do
    file_dir="$results_path/${relative_path}"
    ${collect_test_file} ${file_dir} ${filter_target}
done



