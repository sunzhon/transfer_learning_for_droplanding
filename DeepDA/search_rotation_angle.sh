#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

feature_layer_num=5 # keep it to use five. it is the best value
sub_num=15
rot_id=0
if true; then
    for rot_angle in 130 150 170; do
        for train_sub_num in 14; do
            for trial_num in 25; do
                rot_id=`expr $rot_id + $rot_angle`
                rot_id=$rot_angle

                if false; then
                    model_name="baseline_mlnn"
                    tre_data_relative_path="selection"
                else
                    model_name="augmentation"
                    tre_data_relative_path="augmentation"
                fi

                config_id=${rot_angle}
                tst_data_relative_path="selection"
                tst_data_prefix_name="${sub_num}sub_${trial_num}tri"
                tre_data_prefix_name="${rot_id}rotid_${sub_num}sub_${trial_num}tri"
                base_name="kem_norm_landing_data.hdf5"

                relative_result_folder="${model_name}_v17"
                result_category_folder="${rot_id}rotid/${sub_num}sub/${trial_num}tri"


                #train_sub_num=$(expr $sub_num - $test_sub_num )
                test_sub_num=2
                cv_num=8  #$train_sub_num


                echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"
                log_folder="./log/${model_name}"
                if [ ! -d ${log_folder} ]; then
                    mkdir ${log_folder}
                fi

                python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 0 --yrot_angle 0 --zrot_angle $rot_angle --rot_id "${rot_id}"

                python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${tre_data_relative_path}_${tre_data_prefix_name}_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${tst_data_prefix_name}_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${tre_data_prefix_name}_${config_id}" --config_id ${config_id} --relative_result_folder "${relative_result_folder}/${result_category_folder}" | tee "./log/${model_name}/${tre_dat_prefix_name}.log"
            done
        done
        ./../batch_collect_test_files.sh "${relative_result_folder}/${rot_id}rotid"
    done
else
    for rot_id in $(seq 5 5 30); do
        relative_result_folder="augmentation_v15"
        ./../batch_collect_test_files.sh "${relative_result_folder}/${rot_id}rotid"
    done

fi

