#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
DATA_PATH=`python -c 'import main; print(main.const.DATA_PATH)'`
#features_name=`python -c 'import main; array=main.const.SELECTED_FEATURES_NAME; print(" ".join(array))'`
features_name=`python -c 'import main; array=["Weight","Height"]+main.const.extract_imu_fields(["R_SHANK", "R_THIGH", "R_FOOT", "WAIST", "CHEST", "L_FOOT", "L_SHANK", "L_THIGH"], main.const.ACC_GYRO_FIELDS); print(" ".join(array))'`
feature_layer_num=5 # keep it to use five for offline mode. it is the best value
sub_num=15
for rot_angle in 6 ; do
    for train_sub_num in 14; do
        for trial_num in 25; do #10 15 20 25; do # NOTE: trial_num is only work for base_trail idx, 01, 02,.., not work on 01_0, 01_1,...
            for labels_name in "R_KNEE_MOMENT_X"; do # "R_KNEE_ANGLE_X" ; do
                for model_name in  "augmentation"; do
                    landing_manner="double_leg"
                    config_name="${model_name}_${landing_manner}_${labels_name}_${rot_angle}_${train_sub_num}_${trial_num}"
                    result_folder="${config_name}_t1"

                    echo "features_name: ${features_name}"
                    echo "labels_name: ${labels_name}"
                    echo "train ${model_name} with ${train_sub_num} subjects and ${trial_num} trials"


                    if [[ ${model_name} == "augmentation" ]]; then
                        data_folder="augmentation"
                    else
                        data_folder="selection"
                    fi


                    tre_datafile_basename="tre_${landing_manner}_norm_landing_data.hdf5"
                    tst_datafile_basename="tst_${landing_manner}_norm_landing_data.hdf5"
                    scaler_basename="${landing_manner}_landing_scaler_file.pkl"

                    echo "tre_datafile_basename: ${tre_datafile_basename}"
                    echo "tst_datafile_basename: ${tst_datafile_basename}"
                    echo "scaler file: ${scaler_basename}"


                    #test_sub_num=`expr ${sub_num} - ${train_sub_num}` 
                    test_sub_num=1
                    cv_num=15  # cross validation num

                    # create folder to save log
                    log_folder="./log/${model_name}"
                    if [ ! -d ${log_folder} ]; then
                        mkdir ${log_folder}
                    fi

                    # generate selection dataset file if not exist
                    data_file="${DATA_PATH}/${data_folder}/${tre_datafile_basename}"
                    if [ ! -f ${data_file} ]; then
                        echo $data_file
                        echo "generate selection data ...."
                       # python ./../vicon_imu_data_process/process_landing_data.py
                    fi

                    # generate augmented dataset file if not exist
                    if [[ ${model_name} == "augmentation" ]]; then
                        augmented_filename="${DATA_PATH}/${data_folder}/${tre_datafile_basename}"
                        if [ ! -f  ${augmented_filename} ]; then
                            echo "augment data...."
                          #  python ./../vicon_imu_data_process/augmentation_data.py --xrot_angle 1 --yrot_angle 1 --zrot_angle ${rot_angle} --base_name "${datafile_basename}"  --noise_level 0.1 --scale_level 1.2
                        fi
                    fi

                    # model training and evluation
                    python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${data_folder}/${tre_datafile_basename}" --tst_domain "${data_folder}/${tst_datafile_basename}"  --scaler_file "${data_folder}/${scaler_basename}"  --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${config_name}" --result_folder ${result_folder} --features_name ${features_name} --labels_name ${labels_name} --landing_manner ${landing_manner} | tee "${log_folder}/${config_name}.log"

                   # collect training and test results
                   #./../batch_collect_test_files.sh ${result_folder}
                   array[${result_folder}]=${result_folder}
                done
            done
        done
    done
done


for i in ${array[@]}; do
echo ${array[$i]}
done
