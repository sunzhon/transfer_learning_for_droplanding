#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
DATA_PATH=`python -c 'import main; print(main.const.DATA_PATH)'`
#features_name=`python -c 'import main; array=main.const.SELECTED_FEATURES_NAME; print(" ".join(array))'`
features_name=`python -c 'import main; array=main.const.extract_imu_fields(["R_SHANK", "R_THIGH", "R_FOOT", "WAIST", "CHEST", "L_FOOT", "L_SHANK", "L_THIGH"], main.const.ACC_GYRO_FIELDS); print(" ".join(array))'`
echo "features_name: $features_name"
feature_layer_num=5 # keep it to use five for offline mode. it is the best value
sub_num=15
for rot_angle in 6 ; do
    for train_sub_num in 12; do
        for trial_num in 200; do #10 15 20 25; do
            for labels_name in "R_KNEE_MOMENT_X"; do # "R_KNEE_ANGLE_X" ; do
                for model_name in  "augmentation"; do # "augmentation"; do

                    landing_manner="double_leg"
                    relative_result_folder="${model_name}_t5"
                    echo $labels_name

                    #rot_id=`expr $rot_id + $rot_angle`
                    rot_id=$rot_angle

                    if [[ ${model_name} == "augmentation" ]]; then
                        tre_data_relative_path="augmentation"
                        tre_data_prefix_name="${rot_id}rotid_${landing_manner}"
                    else
                        tre_data_relative_path="selection"
                        tre_data_prefix_name="${landing_manner}"
                    fi

                    tst_data_relative_path="selection"
                    tst_data_prefix_name="${landing_manner}" #NOTE
                    config_id=${rot_angle}

                    datafile_basename="norm_landing_data.hdf5"
                    scaler_basename="landing_scaler_file.pkl"
                    echo $datafile_basename
                    echo $labels_name

                    result_category_folder="${rot_id}rotid/${train_sub_num}sub/${trial_num}tri"

                    #train_sub_num=$(expr $sub_num - $test_sub_num )
                    test_sub_num=1 # if have
                    cv_num=10  #$train_sub_num

                    # create folder to save log
                    echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"
                    log_folder="./log/${model_name}"
                    if [ ! -d ${log_folder} ]; then
                        mkdir ${log_folder}
                    fi

                    # generate selection dataset file if not exist
                    selected_filename="${DATA_PATH}/selection/${landing_manner}_${datafile_basename}"
                    if [ ! -f ${selected_filename} ]; then
                        echo $selected_filename
                        echo "generate selection data ...."
                        python ./../vicon_imu_data_process/process_landing_data.py
                    fi
                    # generate augmented dataset file if not exist

                    if [[ ${model_name} == "augmentation" ]]; then
                        augmented_filename="${DATA_PATH}/${tre_data_relative_path}/${tre_data_prefix_name}_${datafile_basename}"
                        if [ ! -f  ${augmented_filename} ]; then
                            echo "augment data...."
                            python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 1 --yrot_angle 1 --zrot_angle $rot_angle --base_name "${landing_manner}_${datafile_basename}"  --noise_level 0.1 --scale_level 1.2
                        fi
                    fi

                    # model training and evluation
                    python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${tre_data_relative_path}/${tre_data_prefix_name}_${datafile_basename}" --tst_domain "${tst_data_relative_path}/${tst_data_prefix_name}_${datafile_basename}"  --scaler_file "${tst_data_relative_path}/${tst_data_prefix_name}_${scaler_basename}"  --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${tre_data_prefix_name}_${config_id}" --config_id ${config_id} --relative_result_folder "${relative_result_folder}/${result_category_folder}" --features_name ${features_name} --labels_name ${labels_name} --landing_manner ${landing_manner} | tee "${log_folder}/${tre_dat_prefix_name}.log"
                done
            done
        done
    done
    # collect training and test results
    #./../batch_collect_test_files.sh "${relative_result_folder}"
done




