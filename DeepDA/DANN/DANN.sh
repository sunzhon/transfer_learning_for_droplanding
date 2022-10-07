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
for idx in $(seq 2 4 6 8 10 12 14); do
    echo "5 landing trial baseline idx: ${idx}"
    python main.py --config DANN/DANN.yaml --model_selection baseline --tre_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_baseline_v${idx}" --investigation_results_folder "investigation_5trials_baseline_v2/5trials_baseline_v${idx}" | tee "./log/5trials_baseline_${idx}.log"
done

# IMU DATA Augmentation
#for idx in $(seq 6 6); do
#    echo "DANN idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --tgt_domain "augment_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "imu_augment_v${idx}" --investigation_results_folder "imu_augment_v${idx}" | tee "./log/5trials_imu_augment_${idx}.log"
#done


#for idx in $(seq 2 10); do
#    echo " 5trials imu augmentation idx: ${idx}"
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --tgt_domain "augment_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_imu_augment_v${idx}" --investigation_results_folder "investigation_5trials_imu_augment/5trials_imu_augment_v${idx}/" | tee "./log/5trials_imu_augment_${idx}.log"
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


for idx in $(seq 3 4); do
    echo " 5 trial normal DANN idx: ${idx}"
    #python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --tgt_domain "repeated_5trials_{idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_normal_dann_v${idx}" --investigation_results_folder "investigation_5trials_normal_dann_v2/5trials_normal_dann_v${idx}" | tee "./log/5trials_normal_DANN_${idx}.log"
    python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --tcl_domain "repeated_kam_norm_landing_data.hdf5" --tre_domain "repeated_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --tst_domain "repeated_5trials_${idx}subjects_kam_norm_landing_data.hdf5" --config_alias_name "5trials_normal_dann_v${idx}" --investigation_results_folder "investigation_5trials_normal_dann_v2/5trials_normal_dann_v${idx}" | tee "./log/5trials_normal_DANN_${idx}.log"
done
