#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

feature_layer_num=5 # keep it to use five. it is the best value
sub_num=15
rot_id=0
if true; then
    for train_sub_num in 14; do
        for trial_num in 25; do
            for rot_angle in $(seq 5 5 30); do
                rot_id=`expr $rot_id + $rot_angle`
                rot_id=$rot_angle

                if false; then
                    model_name="baseline_mlnn"
                    tre_data_relative_path="selection"
                else
                    model_name="augmentation"
                    tre_data_relative_path="augmentation"
                fi

                config_id="v14"
                tst_data_relative_path="selection"
                base_name="kem_norm_landing_data.hdf5"

                relative_result_folder="${model_name}_${config_id}"

                #train_sub_num=$(expr $sub_num - $test_sub_num )
                test_sub_num=2
                cv_num=8  #$train_sub_num


                echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"

                python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 0 --yrot_angle 0 --zrot_angle $rot_angle --rot_id "${rot_id}"

                python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${rot_id}rotid_${trial_num}trials_${sub_num}subjects_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${train_sub_num}sub_${trial_num}trials_${model_name}_${config_id}" --relative_result_folder "${relative_result_folder}/${rot_id}rotid/${train_sub_num}sub/${trial_num}trials" | tee "./log/${model_name}/${sub_num}sub_${trial_num}trials.log"

                ./../batch_collect_test_files.sh "${relative_result_folder}/${rot_id}rotid"
            done

        done
    done
else
    relative_result_folder="baseline_t1_5"

fi



id"

