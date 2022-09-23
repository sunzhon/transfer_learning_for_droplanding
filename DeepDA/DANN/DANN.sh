#!/usr/bin/env bash

# DATA augmentation
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 2_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 2_5d_imu_augment --investigation_results_folder 2_5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 5d_imu_augment --investigation_results_folder 5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 7_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 7_5d_imu_augment --investigation_results_folder 7_5d_imu_augment | tee imu_augment.log
#
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 10d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 10d_imu_augment --investigation_results_folder 10d_imu_augment | tee imu_augment.log

#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 12_5d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 12_5d_imu_augment --investigation_results_folder 12_5d_imu_augment | tee imu_augment.log

#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain 15d_augment_kam_norm_landing_data.hdf5 --tst_domain kam_norm_landing_data.hdf5 --config_alias_name 15d_imu_augment --investigation_results_folder 15d_imu_augment | tee imu_augment.log




# baseline
#python main.py --config DANN/DANN.yaml --model_selection baseline --n_epoch 100 --tgt_domain 4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5  --config_alias_name baseline_v14 --investigation_results_folder baseline_v14 | tee baseline.log

# IMU DATA Augmentation
#python main.py --config DANN/DANN.yaml --model_selection imu_augment --n_epoch 100 --tgt_domain augment_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name imu_augment_v3 --investigation_results_folder imu_augment_v3 | tee imu_augment.log

# pretrain and fine_tuning
python main.py --config DANN/DANN.yaml --model_selection pretrained --tgt_domain kam_norm_walking_data.hdf5 --config_alias_name preraind_v2 --investigation_results_folder pretrained_v2 --n_epoch 100 --config_alias_name pretrained | tee pretrained.log
#python main.py --config DANN/DANN.yaml --model_selection finetuning --n_epoch 100 --trained_model_state training_testing/model_comparison/pretrained_v2/training_070049  --tgt_domain 4subjects_kam_norm_landing_data.hdf5  --investigation_results_folder finetuning_v1 --config_alias_name finetuning_v1  | tee finetuning.log


# repeated DANN
#rm -r /home/sun/drop_landing_workspace/results/training_tesing/repeated_dann_v4
#python main.py --config DANN/DANN.yaml --model_selection DANN --n_epoch 100 --tgt_domain repeated_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name repeated_dann_v4 --investigation_results_folder repeated_dann_v4 | tee DANN.log

# Aug_DANN
#python main.py --config DANN/DANN.yaml --model_selection Aug_DANN --n_epoch 100 --tgt_domain augment_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name aug_dann_v11  --investigation_results_folder aug_dann_v11 --config_comments 4subjects | tee Aug_DANN.log
#python main.py --config DANN/DANN.yaml --model_selection Aug_DANN --n_epoch 100 --tgt_domain 2_5d_augment_kam_norm_landing_data.hdf5 --tst_domain 2_5d_augment_kam_norm_landing_data.hdf5 --config_alias_name allsub_aug_dann_v11  --investigation_results_folder allsub_aug_dann_v11 --config_comments all_subjects | tee Aug_DANN.log


# Normal_DANN
#python main.py --config DANN/DANN.yaml --model_selection Normal_DANN --n_epoch 100 --tgt_domain augment_4subjects_kam_norm_landing_data.hdf5 --tst_domain 4subjects_kam_norm_landing_data.hdf5 --config_alias_name normal_dann_v3  --investigation_results_folder normal_dann_v3 | tee Normal_DANN.log



