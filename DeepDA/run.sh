#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

for trial_num in 25 ; do
    #for sub_num in 15; do
    sub_num=15
    for feature_layer_num in 10; do
        config_id="v6_${feature_layer_num}"
        model_name="baseline"
        tre_data_relative_path="selection"
        tst_data_relative_path="selection"
        relative_result_folder="baseline_${config_id}"
        base_name="kem_norm_landing_data.hdf5"
        #train_sub_num=`expr $sub_num - 1`
        train_sub_num=14

        #model_name="augmentation"
        #tre_data_relative_path="augmentation"
        #tst_data_relative_path="selection"
        #relative_result_folder="augmentation_v5"
        #base_name="kem_norm_landing_data.hdf5"
        ##train_sub_num=`expr $sub_num - 1`
        #train_sub_num=14
        #config_id=v5



        echo "train ${model_name} with ${sub_num} subjects and ${trial_num} trials"

        python main.py --config run.yaml --model_selection ${model_name} --feature_layer_num ${feature_layer_num} --tre_domain "${tre_data_relative_path}/${tre_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --tst_domain "${tst_data_relative_path}/${tst_data_relative_path}_${trial_num}trials_${sub_num}subjects_${base_name}" --sub_num ${sub_num} --trial_num ${trial_num}  --train_sub_num "${train_sub_num}" --config_name "${sub_num}sub_${trial_num}trials_${model_name}_${config_id}" --relative_result_folder "${relative_result_folder}/${trial_num}trials/${sub_num}sub" | tee "./log/${model_name}/${sub_num}sub_${trial_num}trials.log"
    done
done

