#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

feature_layer_num=5 # keep it to use five. it is the best value
sub_num=15
train_sub_num=14
trial_num=25
for_num=0
if true; then
    for rotation_angle in $(seq -10 -10); do
        for_num=`expr $for_num + 1`

            if false; then
                model_name="baseline_mlnn"
                tre_data_relative_path="selection"
            else
                model_name="augmentation"
                tre_data_relative_path="augmentation"
            fi

            config_id="v12_${rotation_angle}"
            tst_data_relative_path="selection"
            relative_result_folder="${model_name}_${config_id}"
            base_name="kem_norm_landing_data.hdf5"
            test_sub_num=2
            #train_sub_num=$(expr $sub_num - $test_sub_num )
            cv_num=8  #$train_sub_num
            prefix_save_name="${for_num}_augmentation"

            echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"

            python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 0 --yrot_angle 0 --zrot_angle $rotation_angle --prefix_save_name ${prefix_save_name}

            python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${prefix_save_name}_${trial_num}trials_${sub_num}subjects_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${train_sub_num}sub_${trial_num}trials_${model_name}_${config_id}" --relative_result_folder "${relative_result_folder}/${train_sub_num}sub/${trial_num}trials" | tee "./log/${model_name}/${sub_num}sub_${trial_num}trials.log"
        done
else
    relative_result_folder="baseline_t1_5"

fi

./../batch_collect_test_files.sh "${relative_result_folder}"

