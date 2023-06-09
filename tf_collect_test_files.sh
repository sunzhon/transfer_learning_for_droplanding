#! /bin/zsh

# argument
if [ $# -gt 0 ]; then
    testing_folders=$1
else
    testing_folders='.'
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
echo "model_selection\talias_name\tsubject_num\ttrial_num\tlabels_name\tr2\tr_rmse\ttest_subject\tparent_test_id\tchild_test_id\ttraining_testing_folders" > $data_file


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
        labels_name=$(awk -F"[ :-]+" '$2~/R_KNEE_MOMENT_X|R_GRF_Z/{print $2}' $hyper_file)
        test_subject=$(awk -F"[ :-]+" '$1~/test_subject/{print $2}' $hyper_file)
        r2=$(awk -F"[,:]+" '$2~/r2/{print $4}' "${folder_path}/test_metrics.csv")
        r_rmse=$(awk -F"[,:]+" '$2~/r_rmse/{print $4}' "${folder_path}/test_metrics.csv")
        alias_name=$(awk -F"[,:]+" '$1~/config_alias_name/{print $2}' $hyper_file)
        sub_num=${"${alias_name}"%%sub*}
        echo "alsia: ${alias_name}"
        trial_num=${"${"${alias_name}"#*_}"%%trials*}
        echo "sub_num: ${sub_num}, trial_num: ${trial_num}"
        
        if [[ "$alias_name" == "" ]]; then
            alias_name=$model_selection
            #sed -i -e '1a\config_alias_name:imu_augment' $hyper_file
            #alias_name=$(awk -F"[,:]+" '$1~/config_alias_name/{print $2}' $hyper_file)
        fi

        echo "model_selection: ${model_selection}" 
        echo "labels_name: ${labels_name}" 
        echo "r2: ${r2}"


        echo "${model_selection}\t${alias_name}\t${sub_num}\t${trial_num}\t${labels_name}\t${r2}\t${r_rmse}\t${test_subject}\t${parent_test_id}\t${child_test_id}\t${folder_path}" >> $data_file
    fi
done


# delete line with \t as begining, since there lines may be wrong
sed -i -e "/^\t/d" ${data_file}


echo "END TO COLLECT TEST DATA"

