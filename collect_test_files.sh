#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    testing_folders=$1
else
    testing_folders="${MEDIA_NAME}/drop_landing_workspace/results/training_testing/baseline_mlnn_t9"
    testing_folders="${MEDIA_NAME}/drop_landing_workspace/results/training_testing/augmentation_t9"
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

# find all hyperparam.yaml
list_hyper_files=($(find $testing_folders -name hyperparams.yaml))

# create file for store the testing file collections
data_file="$testing_folders/testing_result_folders.txt"
touch ${data_file}
# columns of the testing_result_folders.txt
echo "model_selection\talias_name\tconfig_name\tconfig_id\tsubject_num\ttrial_num\ttrain_sub_num\tfeatures_name\tlabels_name\tr2\trmse\tr_rmse\tmae\ttest_subject\tparent_test_id\tchild_test_id\tlanding_manner\trelative_result_folder\ttraining_testing_folders" > $data_file


echo "START TO COLLECT TEST DATA"


for hyper_file in ${list_hyper_files}; do 
    hyper_file_path=${hyper_file%/*}
    child_test_id=${hyper_file_path##*/}
    parent_test_id=${${hyper_file_path%/*}##*/}
    echo ${hyper_file_path}
    echo ${child_test_id}
    echo ${filter_target}
    echo ${parent_test_id}

    scores_file='w'

    if [[ $child_test_id =~ ${filter_target}[0-9]+ ]];then # list training_* or testing_*
        echo ${hyper_file}
        folder_path=$(cd $(dirname $hyper_file); pwd)
        echo ${folder_path}

        # fields in hyperparams.yaml
        # modify the fields in hyperparams.yaml
        #sed -i -e 's/config_align_name/config_alias_name/g' $hyper_file
        #sed -i -e 's/imu_augment_5degree/complex_baseline/' $hyper_file
        #sed -i -e 's/dann_6/augment_dann/' $hyper_file
        #sed -i -e 's/dann_5/repeated_dann/' $hyper_file
        model_selection=$(awk -F"[ :-]+" '$1~/model_selection/{print $2}' $hyper_file)
        #model_selection=$(awk -F"[ :-]+" '$1~/model_selection/{print $2}' $hyper_file)
        #
        #

        features_name=$(awk -F"[ :-]+" 'BEGIN{flag=0} {if ($1 == "features_name") {flag=1; next}; if ($1 == "labels_name") {flag=0; next}; if(flag == 1) {array[$2]=$2}}END{for (ii in array) print ii}' $hyper_file)
        features_name=$(echo $features_name | tr  '\n' ' ')

        echo "features_names:  ${features_name}"


        labels_name=$(awk -F"[ :-]+" 'BEGIN{flag=0}{{if (flag == 1) print $2; flag=0} {if( $1 == "labels_name" ) flag=1} }' $hyper_file) 
        labels_name=$(echo $labels_name | tr '\n' ' ')
        echo "labels_name: ${labels_name}"

        test_subject=$(awk -F"[ :-]+" '$1~/test_subject/{print $2}' $hyper_file)
        r2=$(awk -F"[,:]+" '$2~/r2/{print $4}' "${folder_path}/test_metrics.csv")
        r_rmse=$(awk -F"[,:]+" '{if( $2 == "r_rmse") {print $4}}' "${folder_path}/test_metrics.csv")
        rmse=$(awk -F"[,:]+" '{if( $2 == "rmse") {print $4}}' "${folder_path}/test_metrics.csv")
        mae=$(awk -F"[,:]+" '$2~/mae/{print $4}' "${folder_path}/test_metrics.csv")

        echo "alias: ${alias_name}"
        sub_num=$(awk -F"[ :-]+" '$1~/^sub_num/{print $2}' $hyper_file)
        trial_num=$(awk -F"[ :-]+" '$1~/^trial_num/{print $2}' $hyper_file)
        train_sub_num=$(awk -F"[ :-]+" '$1~/^train_sub_num/{print $2}' $hyper_file)
        echo "sub_num: ${sub_num}, trial_num: ${trial_num}, train_sub_num: ${train_sub_num}"
        
        config_name=$(awk -F"[,:]+" '$1~/config_name/{print $2}' $hyper_file)
        config_id=$(awk -F"[,:]+" '$1~/config_id/{print $2}' $hyper_file)
        relative_result_folder=$(awk -F"[,:]+" '$1~/relative_result_folder/{print $2}' $hyper_file)

        landing_manner=$(awk -F"[,:]+" '$1~/^landing_manner/{print $2}' $hyper_file)

        alias_name=$(awk -F"[,:]+" '$1~/^alias_name/{print $2}' $hyper_file)
        #alias_name=$(model_selection#"trials_")
        if [[ "$alias_name" == "" ]]; then
            alias_name=$model_selection
            #sed -i -e '1a\config_alias_name:imu_augment' $hyper_file
            #alias_name=$(awk -F"[,:]+" '$1~/config_alias_name/{print $2}' $hyper_file)
        fi

        echo "model_selection: ${model_selection}" 
        echo "r2: ${r2}, rmse: ${rmse}, r_rmse: ${r_rmse}, mae: ${mae}"


        echo "${model_selection}\t${alias_name}\t${config_id}\t${config_name}\t${sub_num}\t${trial_num}\t${train_sub_num}\t${features_name}\t${labels_name}\t${r2}\t${rmse}\t${r_rmse}\t${mae}\t${test_subject}\t${parent_test_id}\t${child_test_id}\t${landing_manner}\t${relative_result_folder}\t${folder_path}" >> $data_file
    fi
done


# delete line with \t as begining, since there lines may be wrong
sed -i -e "/^\t/d" ${data_file}


echo "END TO COLLECT TEST DATA"

