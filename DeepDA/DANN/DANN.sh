#!/usr/bin/env bash
#!/bin/pyenv python
#coding: --utf-8

# DATA augmentation
#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 2_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 2_5d_imu_augment --investigation_results_folder 2_5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 5d_imu_augment --investigation_results_folder 5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 7_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 7_5d_imu_augment --investigation_results_folder 7_5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 10d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 10d_imu_augment --investigation_results_folder 10d_imu_augment | tee imu_augment.log

#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 12_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 12_5d_imu_augment --investigation_results_folder 12_5d_imu_augment | tee imu_augment.log

#python main.py --config DANN/DANN.yaml --model_selection imu_augment  --tgt_domain 15d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 15d_imu_augment --investigation_results_folder 15d_imu_augment | tee imu_augment.log


# baseline
#for sub_idx in $(seq 13 14); do
##for sub_idx in 2 4 6 8 10 12 14; do
#    #for trial_idx in 5 10 15 20 25; do
#    for trial_idx in 25; do
#        echo "${trial_idx} landing trial baseline idx: ${sub_idx}"
#        python main.py --config DANN/DANN.yaml --model_selection baseline --n_epoch 200 --tre_domain "selected_data/${trial_idx}trials_${sub_idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "selected_data/${trial_idx}trials_${sub_idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "${sub_idx}sub_${trial_idx}trials_baseline_v${sub_idx}" --investigation_results_folder "investigation_baseline_v1/${trial_idx}trials/${sub_idx}sub" | tee "./log/${sub_idx}sub_${trial_idx}trials_baseline.log"
#    done
#done


for train_trial_num in 25; do
    for train_sub_num in $(seq 1 14); do
        echo "baseline train with ${train_sub_num} subjects and ${train_trial_num} trials"
        python main.py --config DANN/DANN.yaml --model_selection baseline --n_epoch 200 --online_select_dataset "selected_data/${train_trial_num}trials_15subjects_kam_norm_landing_data.hdf5" --train_sub_num ${train_sub_num} --train_trial_num ${train_trial_num}  --config_alias_name "${train_sub_num}sub_${train_trial_num}trials_baseline_v${train_sub_num}" --investigation_results_folder "investigation_baseline_v2/${train_trial_num}trials/${train_sub_num}sub" | tee "./log/${train_sub_num}sub_${train_trial_num}trials_baseline.log"
    done
done


# IMU DATA Augmentation
#for idx in $(seq 6 6); do
#    echo "DANN idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --tgt_domain "augment_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "imu_augment_v${idx}" --investigation_results_folder "imu_augment_v${idx}" | tee "./log/5trials_imu_augment_${idx}.log"
#done


#for sub_idx in $(seq 5 11); do
#for sub_idx in 5 7 9 11; do
#    for trial_idx in 15 20 25; do
#        echo " ${trial_idx}trials imu augmentation idx: ${sub_idx}"
#        python main.py --config DANN/DANN.yaml --model_selection imu_augment --tre_domain "augment_data/augment_${trial_idx}trials_${sub_idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "selected_data/${trial_idx}trials_${sub_idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "${sub_idx}sub_${trial_idx}trials_baseline_v${sub_idx}" --investigation_results_folder "investigation_imu_augment_v1/${trial_idx}trials/${sub_idx}sub" | tee "./log/${sub_idx}sub_${trial_idx}trials_imu_augment.log"
#    done
#done


# pretrain and fine_tuning
#python main.py --config DANN/DANN.yaml --model_selection pretrained --tgt_domain kam_norm_walking_data.hdf5 --config_alias_name preraind_v2 --investigation_results_folder walking_model/pretrained_v2  --config_alias_name pretrained_v2  --n_split 5 | tee log/pretrained_v2.log

#for idx in $(seq 2 10); do
#    echo "5trials finetuning idx: ${idx}"
#    python main.py --config DANN/DANN.yaml --model_selection finetuning  --trained_model_state "training_testing/walking_model/pretrained_v1/training_130423" --tgt_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_finetuning_v${idx}" --investigation_results_folder "investigation_5trials_finetuning/5trials_finetuning_v${idx}" | tee "5trials_finetuning_${idx}.log"
#done


# only source domain for train
#for idx in $(seq 6 6); do
#    echo "10trials test_model idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection pretrained --tgt_domain kam_norm_walking_data.hdf5 --config_alias_name preraind_v1 --investigation_results_folder walking_model/pretrained_v1  --config_alias_name pretrained_v1  | tee log/pretrained_v3.log
#    python main.py --config DANN/DANN.yaml --model_selection test_model  --trained_model_state "training_testing/walking_model/pretrained_v1/training_130423" --tgt_domain kam_norm_walking_data.hdf5 --tst_domain "10trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "10trials_test_model_v${idx}" --investigation_results_folder "investigation_10trials_test_model/10trials_test_model_v${idx}" | tee "10trials_test_model_${idx}.log"
#done


# repeated DANN
#rm -r /home/sun/drop_landing_workspace/results/training_testing/repeated_dann_v4
#for idx in $(seq 3 4); do
#    echo "5 trial repeated DANN idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection DANN --tgt_domain "repeated_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_repeated_dann_v${idx}" --investigation_results_folder "investigation_5trials_repeated_dann/5trials_repeated_dann_v${idx}" | tee "./log/5trials_repeated_DANN_${idx}.log"
#done

# Aug_DANN
#python main.py --config DANN/DANN.yaml --model_selection Aug_DANN  --tgt_domain augment_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name aug_dann_v11  --investigation_results_folder aug_dann_v11 --config_comments 4subjects | tee Aug_DANN.log

#for idx in $(seq 6 6); do
#    echo "Augmentation DANN idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection Aug_DANN --tgt_domain "augment_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "augment_dann_v${idx}" --investigation_results_folder "augment_dann_v${idx}" | tee "Aug_DANN_${idx}.log"
#done

#for idx in $(seq 2 6); do
#    echo "5 trials' Augmentation DANN idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection Aug_DANN --tgt_domain "augment_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_augment_dann_v${idx}" --investigation_results_folder "investigation_5trials_augment_dann_v2/5trials_augment_dann_v${idx}" | tee "./log/5trials_Aug_DANN_${idx}.log"
#done


# Normal_DANN
#python main.py --config DANN/DANN.yaml --model_selection Normal_DANN  --tgt_domain augment_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name normal_dann_v3  --investigation_results_folder normal_dann_v3 | tee Normal_DANN.log

#for idx in $(seq 12 12); do
#    echo "10 trial normal DANN idx: ${idx}"
#    python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --tgt_domain "repeated_10trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "10trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "10trials_normal_dann_v${idx}" --investigation_results_folder "investigation_10trials_normal_dann/10trials_normal_dann_v${idx}" | tee "10trials_normal_DANN_${idx}.log"
#done


#for idx in $(seq 3 4); do
#    echo " 5 trial normal DANN idx: ${idx}"
#    #python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --tgt_domain "repeated_5trials_{idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_normal_dann_v${idx}" --investigation_results_folder "investigation_5trials_normal_dann_v2/5trials_normal_dann_v${idx}" | tee "./log/5trials_normal_DANN_${idx}.log"
#    python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --tcl_domain "repeated_kam_norm_landing_data.hdf5" --tre_domain "repeated_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "repeated_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_normal_dann_v${idx}" --investigation_results_folder "investigation_5trials_normal_dann_v2/5trials_normal_dann_v${idx}" | tee "./log/5trials_normal_DANN_${idx}.log"
#done

# inter-subject normal DANN
#rm -r /home/sun/drop_landing_workspace/results/training_testing/investigation_inersub_normal_dann_v1
#for sub_idx in '08' '10' '11' '14' '15'; do
#for sub_idx in '16' '17' '18' '19' '20' '21' '22' '23' '24'; do
#    echo "inter sub normal DANN idx: ${sub_idx}"
#    python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --batch_size 20 --src_domain "intersub/sub${sub_idx}_src_kam_norm_landing_data.hdf5" --tcl_domain "intersub/sub${sub_idx}_tcl_kam_norm_landing_data.hdf5" --tre_domain "intersub/sub${sub_idx}_tst_kam_norm_landing_data.hdf5" --tst_domain "intersub/sub${sub_idx}_tst_kam_norm_landing_data.hdf5" --config_alias_name "intersub_dann_${sub_idx}sub_normal_dann_v${idx}" --investigation_results_folder "investigation_intersub_normal_dann_v1/${sub_idx}sub" | tee "./log/${sub_idx}_intersub_normal_DANN.log"
#
#done

