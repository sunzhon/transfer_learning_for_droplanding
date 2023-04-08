#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8
DATA_PATH=`python -c 'import main; print(main.const.DATA_PATH)'`
#features_name=`python -c 'import main; array=main.const.SELECTED_FEATURES_NAME; print(" ".join(array))'`
features_name=`python -c 'import main; array=["Weight","Height"]+main.const.extract_imu_fields(["R_SHANK", "R_THIGH", "R_FOOT", "WAIST", "CHEST", "L_FOOT", "L_SHANK", "L_THIGH"], main.const.ACC_GYRO_FIELDS); print(" ".join(array))'`
feature_layer_num=5 # keep it to use five for offline mode. it is the best value
sub_num=17
cv_num=15  # cross validation num
result_folder_array=()
for rot_angle in 6 ; do
    for train_sub_num in 3 7 9; do
        test_sub_num=`expr ${sub_num} - ${train_sub_num}` 
        for trial_num in 30; do #10 15 20 25; do # NOTE: trial_num is only work for base_trail idx, 01, 02,.., not work on 01_0, 01_1,...
            for labels_name in "R_KNEE_MOMENT_X"; do # "R_KNEE_ANGLE_X" ; do
                for model_name in  "baseline"; do # "augmentation"; do
                    for dataset_name in "original" "rotation" "timewarp"; do #"timewarp"; do #"original" "rotation"; do #"rotation"; do # "rotation" "time_wrap"; do
                        landing_manner="rdouble_leg"
                        config_name="${model_name}_${landing_manner}_${dataset_name}_${labels_name}_${rot_angle}_${train_sub_num}_${trial_num}"

                        # data_id="a3" #a3 no noise, mean r2=0.75, it is better than using noise.
                        # data_id="a4" #a4 no noise and node scale, mean r2=0.72
                        # data_id="tw1" # tw1, mean r2=0.79
                        data_id="a5" # test_sub_num=14, mean r2 =  0.83 (rotation), 0.77 (original)


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
                        result_folder="${config_name}_${data_id}"

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
                        ./../batch_collect_test_files.sh ${result_folder}
                    done
                done
            done
        done
    done
done

######
echo "This is the result folders: ..."

result_folders="./result_folders.txt"
touch ${result_folders}

for ii in ${result_folder_array[@]}; do
    echo ${ii}
    echo "${ii}" >> ${result_folders}
    #`python -c import assessment.scores as *; metrics = get_investigation_metrics(path)`
done

python ./../assessments/scores.py  "${result_folder_array[@]}"
