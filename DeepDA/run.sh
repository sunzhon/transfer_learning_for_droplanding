#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

feature_layer_num=5 # keep it to use five. it is the best value
sub_num=15
if true; then
    for train_sub_num in $(seq 1 14); do
        for trial_num in $(seq 25 25); do

            if true; then
                model_name="baseline_mlnn"
                tre_data_relative_path="selection"
            else
                model_name="augmentation"
                tre_data_relative_path="augmentation"
            fi

            config_id="v10_${feature_layer_num}"
            tst_data_relative_path="selection"
            relative_result_folder="${model_name}_${config_id}"
            base_name="kem_norm_landing_data.hdf5"
            test_sub_num=2
            #train_sub_num=$(expr $sub_num - $test_sub_num )
            cv_num=8  #$train_sub_num

            echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"

            python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${tre_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${train_sub_num}sub_${trial_num}trials_${model_name}_${config_id}" --relative_result_folder "${relative_result_folder}/${train_sub_num}sub/${trial_num}trials" | tee "./log/${model_name}/${sub_num}sub_${trial_num}trials.log"
        done
    done

else
    relative_result_folder="baseline_t1_5"

fi

./../batch_collect_test_files.sh "${relative_result_folder}"

