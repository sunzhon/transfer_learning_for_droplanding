#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
DATA_PATH=`python -c 'import main; print(main.const.DATA_PATH)'`
#features_name=`python -c 'import main; array=main.const.SELECTED_FEATURES_NAME; print(" ".join(array))'`
sub_num=15
cv_num=15  # cross validation num
result_folder_array=()
result_folder_file="./result_folders.txt"
tmp_result_folder_file="./tmp_result_folders.txt"
for landing_manner in "rdouble_leg_v1"; do
    for model_name in "baseline"; do # "augmentation"; do
        for feature_layer_num in 1 2 3 4 5 6 7; do # keep it to use five for offline mode. it is the best value
            for dataset_name in "original";  do #"da_rotation"  "e_rotation"; do #"da_rotation" "e_rotation"; do #"da_rotation" "e_rotation"; do #"original" "e_scale" "da_scale"; do #"da_scale" "da_rotation" "e_rotation"; do #"timewarp"; do #"original" "rotation"; do #"rotation"; do # "rotation" "time_wrap"; do
                for train_sub_num in 3; do # 8 9 10 11 12 13 14 15; do
                    test_sub_num=`expr ${sub_num} - ${train_sub_num}` 
                    for tre_trial_num in 15 ; do # NOTE: tre_trial_num is only work for base_trail idx of tst, 01, 02,.., not work on 01_0, 01_1,...
                        for tst_trial_num in 25; do #10 15 20 25; do # NOTE: tst_trial_num is only work for base_trail idx of tst, 01, 02,.., not work on 01_0, 01_1,...
                            for labels_name in "R_KNEE_MOMENT_X"; do # "R_KNEE_ANGLE_X" ; do
                                features_name=`python -c 'import main; array=["Weight","Height"] + main.const.extract_imu_fields(["R_SHANK", "R_THIGH", "R_FOOT", "WAIST", "CHEST", "L_FOOT", "L_SHANK", "L_THIGH"], main.const.ACC_GYRO_FIELDS); print(" ".join(array))'`
                                scale_method="standard"
                                data_id="ori_v2" # test_sub_num=14, mean r2 =  0.83 (rotation), 0.77 (original)
                                dataset_folder=`echo ${dataset_name} | sed -e "s/ /_/g"`
                                config_name="${landing_manner}_${model_name}_${feature_layer_num}_${dataset_folder}_${train_sub_num}_${tre_trial_num}_${tst_trial_num}_${labels_name}_${data_id}"
                                echo "Start to train and test a model ......"
                                echo "dataset_name: ${dataset_name}"
                                echo "train subject num: ${train_sub_num}"
                                echo "test subject num: ${test_sub_num}"
                                echo "features_name: ${features_name}"
                                echo "labels_name: ${labels_name}"
                                echo "dataset_folder: ${dataset_folder}"
                                sleep 3

                                # determine datasets' name from the generated data
                                tre_datafile_basename="tre_${data_id}_${landing_manner}_norm_landing_data.hdf5"
                                tst_datafile_basename="tst_${data_id}_${landing_manner}_norm_landing_data.hdf5"
                                scaler_filename="${data_id}_${landing_manner}_landing_scaler_file.pkl"
                                result_folder="${config_name}"

                                echo "tre_datafile_basename: ${tre_datafile_basename}"
                                echo "tst_datafile_basename: ${tst_datafile_basename}"
                                echo "scaler file: ${scaler_filename}"

                                # create folder to save log
                                log_folder="./log/${model_name}/${dataset_folder}"
                                if [ ! -d ${log_folder} ]; then
                                    mkdir ${log_folder} -p
                                fi

                                # generate dataset file if not exist
                                tre_data_file="${DATA_PATH}/${dataset_folder}/${tre_datafile_basename}"
                                echo "tre_data_file: ${tre_data_file}"
                                if [ ! -f ${tre_data_file} ]; then
                                    echo "${tre_data_file} does not exist, now generate it ...."
                                    python ./../vicon_imu_data_process/process_landing_data.py ${landing_manner} ${scale_method} ${data_id} ${dataset_name} | tee "${log_folder}/data_generattion_${data_id}.log"
                                fi

                                # model training and evluation
                                python main.py --config run.yaml --model_name ${model_name} --dataset_name ${dataset_folder} --feature_layer_num ${feature_layer_num} --cv_num ${cv_num} --tre_domain "${dataset_folder}/${tre_datafile_basename}" --tst_domain "${dataset_folder}/${tst_datafile_basename}"  --scaler_file "${dataset_folder}/${scaler_filename}"  --sub_num ${sub_num} --tst_trial_num ${tst_trial_num} --tre_trial_num ${tre_trial_num} --train_sub_num "${train_sub_num}" --test_sub_num ${test_sub_num} --config_name "${config_name}" --result_folder ${result_folder} --features_name ${features_name} --labels_name ${labels_name} --landing_manner ${landing_manner} | tee "${log_folder}/${config_name}.log"

                                # collect training and test results
                                result_folder_array+=(${result_folder})
                                echo ${result_folder} >> ${result_folder_file}
                                echo ${result_folder} > ${tmp_result_folder_file}
                                ./../batch_collect_test_files.sh ${result_folder}
                            done
                        done
                    done
                done
            done
        done
    done
done

### store resulst folder
cp tmp_result_folders.txt $MEDIA_NAME/drop_landing_workspace/results/training_testing
echo "This is the result folders: ..."
#python ./../assessments/scores.py  "${result_folder_array[@]}"
python calculate_metrics.py  "${result_folder_array[@]}"



