#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

feature_layer_num=5 # keep it to use five. it is the best value
sub_num=15
rot_id=0
#for rot_angle in 10 30 50 70 90 ; do
for rot_angle in 6; do
    for train_sub_num in 8 9 10 11 12 13 14; do
        for trial_num in 5 10 15 20; do #10 15 20 25; do
            for target_variable in "R_KNEE_MOMENT_X"; do
                for model_name in  "augmentation"; do

                    #rot_id=`expr $rot_id + $rot_angle`
                    rot_id=$rot_angle
                    landing_manner="double_legs"

                    if [[ ${model_name} == "baseline_mlnn" ]]; then
                        tre_data_relative_path="selection"
                        tre_data_prefix_name="${sub_num}sub_${trial_num}tri"
                    else
                        tre_data_relative_path="augmentation"
                        tre_data_prefix_name="${rot_id}rotid_${sub_num}sub_${trial_num}tri"
                    fi

                    relative_result_folder="${model_name}_kem_v3"
                    tst_data_relative_path="selection"
                    #tst_data_prefix_name="${sub_num}sub_${trial_num}tri"
                    tst_data_prefix_name="${sub_num}sub_15tri" #NOTE
                    config_id=${rot_angle}

                    if [[ ${landing_manner} == "double_legs" ]]; then
                        if [[ ${target_variable} == "R_KNEE_MOMENT_X" ]]; then
                            base_name="kem_norm_landing_data.hdf5"
                        elif [[ ${target_variable} == "R_KNEE_MOMENT_Y" ]]; then
                            base_name="kam_norm_landing_data.hdf5"
                        elif [[ ${target_variable} == "PEAK_R_KNEE_MOMENT_X" ]]; then
                            base_name="pkem_norm_landing_data.hdf5"
                        elif [[ ${target_variable} == "PEAK_R_KNEE_MOMENT_Y" ]]; then
                            base_name="pkam_norm_landing_data.hdf5"
                        fi
                    elif [[ ${landing_manner} == "single_leg" ]]; then

                        if [[ ${target_variable} == "R_KNEE_MOMENT_X" ]]; then
                            base_name="single_kem_norm_landing_data.hdf5"
                        else
                            base_name="single_kam_norm_landing_data.hdf5"
                        fi
                    fi

                    echo $base_name
                    echo $target_variable

                    result_category_folder="${rot_id}rotid/${train_sub_num}sub/${trial_num}tri"


                    #train_sub_num=$(expr $sub_num - $test_sub_num )
                    test_sub_num=1 # if have
                    cv_num=15  #$train_sub_num


                    echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"
                    log_folder="./log/${model_name}"
                    if [ ! -d ${log_folder} ]; then
                        mkdir ${log_folder}
                    fi

                    # generate dataset file if not exsit
                    if [[ ${model_name} == "augmentation___" ]]; then
                        if [ ! -f "${DATA_PATH}/${tre_data_relative_path}/${tre_data_prefix_name}_${base_name}" ]; then
                            echo "augment data...."
                            python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 1 --yrot_angle 1 --zrot_angle $rot_angle --rot_id "${rot_id}" --sub_num ${sub_num} --tri_num ${trial_num} --base_name ${base_name} --noise_level 0.1 --scale_level 1.2
                        fi
                    fi

                    # model training and evluation
                    python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${tre_data_relative_path}_${tre_data_prefix_name}_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${tst_data_prefix_name}_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${tre_data_prefix_name}_${config_id}" --config_id ${config_id} --relative_result_folder "${relative_result_folder}/${result_category_folder}" --labels_name ${target_variable} --landing_manner ${landing_manner} | tee "./log/${model_name}/${tre_dat_prefix_name}.log"
                done
            done
        done
    done
    # collect training and test results
    ./../batch_collect_test_files.sh "${relative_result_folder}/${rot_id}rotid"
done
