#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
DATA_PATH=`python -c 'import main; print(main.const.DATA_PATH)'`
#features_name=`python -c 'import main; array=main.const.SELECTED_FEATURES_NAME; print(" ".join(array))'`
sub_num=17
cv_num=15  # cross validation num
result_folder_array=()
result_folder_file="./result_folders.txt"
for landing_manner in "rdouble_leg"; do
    for model_name in "baseline"; do # "augmentation"; do
        for feature_layer_num in 1 2 3 4 5 6; do # keep it to use five for offline mode. it is the best value
            #for dataset_name in "original" "rotation" "timewarp"; do #"timewarp"; do #"original" "rotation"; do #"rotation"; do # "rotation" "time_wrap"; do
            for dataset_name in "e_scale" "da_scale"; do #"da_scale" "da_rotation" "e_rotation"; do #"timewarp"; do #"original" "rotation"; do #"rotation"; do # "rotation" "time_wrap"; do
                for train_sub_num in 1 3 5 7 9 11 13 15; do
                    test_sub_num=`expr ${sub_num} - ${train_sub_num}` 
                    for trial_num in 30; do #10 15 20 25; do # NOTE: trial_num is only work for base_trail idx, 01, 02,.., not work on 01_0, 01_1,...
                        for labels_name in "R_KNEE_MOMENT_X"; do # "R_KNEE_ANGLE_X" ; do
                            features_name=`python -c 'import main; array=["Weight","Height"]+main.const.extract_imu_fields(["R_SHANK", "R_THIGH", "R_FOOT", "WAIST", "CHEST", "L_FOOT", "L_SHANK", "L_THIGH"], main.const.ACC_GYRO_FIELDS); print(" ".join(array))'`
                            data_id="a5" # test_sub_num=14, mean r2 =  0.83 (rotation), 0.77 (original)
                            config_name="${landing_manner}_${model_name}_${feature_layer_num}_${dataset_name}_${train_sub_num}_${trial_num}_${labels_name}_${data_id}"
                            # data_id="a3" #a3 no noise, mean r2=0.75, it is better than using noise.
                            # data_id="a4" #a4 no noise and node scale, mean r2=0.72
                            # data_id="tw1" # tw1, mean r2=0.79


                            echo "Start to train and test a model ......"

                            echo "dataset_name: ${dataset_name}"
                            echo "train subject num: ${train_sub_num}"
                            echo "test subject num: ${test_sub_num}"
                            echo "features_name: ${features_name}"
                            echo "labels_name: ${labels_name}"
                            sleep 1
                            #echo "train ${model_name} with ${dataset_name} having ${train_sub_num} subjects and ${trial_num} trials"

                            data_folder="${dataset_name}"
                            tre_datafile_basename="tre_${data_id}_${landing_manner}_norm_landing_data.hdf5"
                            tst_datafile_basename="tst_${data_id}_${landing_manner}_norm_landing_data.hdf5"
                            scaler_basename="${data_id}_${landing_manner}_landing_scaler_file.pkl"
                            result_folder="${config_name}"

                            echo "tre_datafile_basename: ${tre_datafile_basename}"
                            echo "tst_datafile_basename: ${tst_datafile_basename}"
                            echo "scaler file: ${scaler_basename}"



                            # create folder to save log
                            log_folder="./log/${model_name}/${dataset_name}"
                            if [ ! -d ${log_folder} ]; then
                                mkdir ${log_folder}
                            fi

                            # generate dataset file if not exist
                            tre_data_file="${DATA_PATH}/${data_folder}/${tre_datafile_basename}"
                            echo "tre_data_file: ${tre_data_file}"
                            if [ ! -f ${tre_data_file} ]; then
                                echo "${tre_data_file} is not exist, now generate it ...."
                                python ./../vicon_imu_data_process/process_landing_data.py ${landing_manner} ${dataset_name} "minmax" ${data_id}
                            fi

                            # model training and evluation
                            python main.py --config run.yaml --model_name ${model_name} --dataset_name ${dataset_name} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${data_folder}/${tre_datafile_basename}" --tst_domain "${data_folder}/${tst_datafile_basename}"  --scaler_file "${data_folder}/${scaler_basename}"  --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${config_name}" --result_folder ${result_folder} --features_name ${features_name} --labels_name ${labels_name} --landing_manner ${landing_manner} | tee "${log_folder}/${config_name}.log"

                            # collect training and test results
                            result_folder_array+=(${result_folder})
                           echo ${result_folder} >> ${result_folder_file}
                            ./../batch_collect_test_files.sh ${result_folder}
                        done
                    done
                done
            done
        done
    done
done

### store resulst folder
echo "This is the result folders: ..."
python ./../assessments/scores.py  "${result_folder_array[@]}"

